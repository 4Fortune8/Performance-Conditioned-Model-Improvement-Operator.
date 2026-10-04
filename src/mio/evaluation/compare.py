"""Compare several operators evaluated on the same parents (one evaluation run per operator).

Each run evaluates one operator plus the shared baselines on identical parents,
reference curves and evaluation examples, so rows can be merged: baseline rows
come from the first run (and are checked against the others), and every
operator-derived method is prefixed with its run label. All comparisons are
paired per init group; the train-root gap (memorization) is unpaired.
"""

from __future__ import annotations

import numpy as np

from mio.evaluation.analysis import _ci, _fmt, class_specificity, controllability, generalization_gap, summarize

OPERATOR_PREFIXES = ("operator", "select_with_operator")


def is_operator_method(method: str) -> bool:
    return method.startswith(OPERATOR_PREFIXES)


def merge_runs(runs: dict[str, list[dict]], tol: float = 1e-6) -> tuple[list[dict], list[str]]:
    """Merge per-operator runs; returns (rows, warnings about baseline mismatches)."""
    labels = list(runs)
    first = labels[0]
    base_index = {(r["method"], r["source_id"], r["horizon"]): r for r in runs[first] if not is_operator_method(r["method"])}
    warnings = []
    merged = list(base_index.values())
    for label in labels:
        for r in runs[label]:
            if is_operator_method(r["method"]):
                merged.append({**r, "method": f"{label}:{r['method']}"})
            elif label != first:
                ref = base_index.get((r["method"], r["source_id"], r["horizon"]))
                if ref is None:
                    warnings.append(f"{label}: baseline row {r['method']} {r['source_id']} h{r['horizon']} missing in {first}")
                elif abs(ref["gated_gains"]["report"]["loss"] - r["gated_gains"]["report"]["loss"]) > tol:
                    warnings.append(f"{label}: baseline {r['method']} differs from {first} at {r['source_id']} h{r['horizon']}")
    return merged, warnings


def _strip(rows: list[dict], label: str) -> list[dict]:
    """Rows of one run's operator methods under their original names (for controllability etc.)."""
    p = f"{label}:"
    return [{**r, "method": r["method"][len(p):]} for r in rows if r["method"].startswith(p)]


def comparison_report(runs: dict[str, list[dict]], train_runs: dict[str, list[dict]] | None = None,
                      n_boot: int = 2000, seed: int = 0, title: str = "Operator comparison") -> tuple[str, dict]:
    rows, warnings = merge_runs(runs)
    num_classes = len(rows[0]["parent_metrics"]["report"]["class_loss"]) if rows else 0
    show = [m for m in sorted({r["method"] for r in rows})
            if m != "no_update" and "operator_q" not in m and "operator_class_request" not in m]
    vs_none = {(s["method"], s["horizon"]): s for s in summarize(rows, "no_update", n_boot, seed)}
    vs_sel = {(s["method"], s["horizon"]): s for s in summarize(rows, "select_simple", n_boot, seed)}
    lines = [f"# {title}", "",
             "Unit: init group (paired over identical parents). Gains: report-split NLL, positive = better. "
             "`gated` = after accept/rollback on the accept split. Primary comparison: gated gain minus the gated "
             "best-simple-baseline selector `select_simple`.", ""]
    if warnings:
        lines += ["**Baseline mismatches between runs:**", ""] + [f"- {w}" for w in warnings[:20]] + [""]
    out: dict = {"summary_vs_no_update": list(vs_none.values()), "summary_vs_select_simple": list(vs_sel.values())}
    for h in sorted({r["horizon"] for r in rows}):
        lines += [f"## Horizon {h}", "",
                  "| method | groups | ungated gain | 95% CI | gated gain | gated − select_simple | 95% CI | "
                  "P(group > sel) | Δacc | accept rate | cost (fb eq) |",
                  "|" + "---|" * 11]
        ms = [m for m in show if (m, h) in vs_none]
        ms.sort(key=lambda m: -np.nan_to_num(vs_sel.get((m, h), {}).get("gated_diff_vs_ref_mean", np.nan), nan=-9))
        for m in ms:
            a, b = vs_none[(m, h)], vs_sel.get((m, h), {})
            lines.append(
                f"| {m} | {a['n_groups']} | {_fmt(a['gain_mean'])} | {_ci(a['gain_ci'])} | {_fmt(a['gated_gain_mean'])} | "
                f"{_fmt(b.get('gated_diff_vs_ref_mean', np.nan))} | {_ci(b.get('gated_diff_vs_ref_ci', (np.nan, np.nan)))} | "
                f"{_fmt(b.get('p_group_gated_improves_vs_ref', np.nan), 2)} | {_fmt(a['acc_gain_mean'] * 100, 2)} pp | "
                f"{_fmt(a['accept_rate'], 2)} | {_fmt(a['fb_equivalent_mean'], 1)} |")
        lines.append("")

    # ---- operator vs operator (paired, gated and ungated) -------------------
    labels = list(runs)
    if len(labels) > 1:
        ref_label = labels[0]
        lines += [f"## Operators vs `{ref_label}:operator` (paired per group)", "",
                  "| operator | horizon | ungated diff | 95% CI | gated diff | 95% CI | P(group > ref, gated) |",
                  "|---|---|---|---|---|---|---|"]
        cmp_rows = [r for r in rows if r["method"].endswith(":operator")]
        for s in summarize(cmp_rows, f"{ref_label}:operator", n_boot, seed):
            if s["method"] == f"{ref_label}:operator":
                continue
            lines.append(f"| {s['method']} | {s['horizon']} | {_fmt(s['diff_vs_ref_mean'])} | {_ci(s['diff_vs_ref_ci'])} | "
                         f"{_fmt(s['gated_diff_vs_ref_mean'])} | {_ci(s['gated_diff_vs_ref_ci'])} | "
                         f"{_fmt(s['p_group_gated_improves_vs_ref'], 2)} |")
        lines.append("")

    # ---- conditioning (H3) --------------------------------------------------
    lines += ["## Conditioning (H3)", "",
              "| operator | sweep Spearman | 95% CI | class specificity | 95% CI | diag mean | off-diag mean |",
              "|---|---|---|---|---|---|---|"]
    out["conditioning"] = {}
    for label in labels:
        own = _strip(rows, label)
        c = controllability(own, n_boot, seed)
        sp = class_specificity(own, num_classes, n_boot, seed)
        out["conditioning"][label] = {"controllability": c, "class_specificity": sp}
        lines.append(f"| {label} | {_fmt(c.get('spearman_mean'), 3)} | {_ci(c.get('spearman_ci', (np.nan, np.nan)), 3)} | "
                     f"{_fmt(sp.get('specificity_mean', np.nan))} | {_ci(sp.get('specificity_ci', (np.nan, np.nan)))} | "
                     f"{_fmt(sp.get('diag_mean', np.nan))} | {_fmt(sp.get('offdiag_mean', np.nan))} |")
    lines.append("")

    # ---- memorization gap (H2) ----------------------------------------------
    if train_runs:
        train_rows, _ = merge_runs({k: v for k, v in train_runs.items()})
        gaps = generalization_gap(train_rows, rows, n_boot, seed, gated=False)
        out["generalization_gap"] = gaps
        lines += ["## Train roots vs held-out roots (memorization, ungated gain)", "",
                  "Unpaired (different groups). Baselines estimate how much the two root sets differ by chance.", "",
                  "| method | horizon | train-root gain | held-out gain | gap | 95% CI | groups (train / held-out) |",
                  "|---|---|---|---|---|---|---|"]
        for g in gaps:
            if "operator_q" in g["method"]:
                continue
            lines.append(f"| {g['method']} | {g['horizon']} | {_fmt(g['train_mean'])} | {_fmt(g['heldout_mean'])} | "
                         f"{_fmt(g['gap'])} | {_ci(g['gap_ci'])} | {g['n_train_groups']} / {g['n_heldout_groups']} |")
        lines.append("")
    return "\n".join(lines), out

