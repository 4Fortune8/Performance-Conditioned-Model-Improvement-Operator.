"""Run-level statistics for evaluation results.

The statistical unit is the *init group* (independent initialization), never
the individual checkpoint: rows are first averaged within each group, then
uncertainty is estimated by bootstrapping over groups. Method comparisons are
paired (same parent checkpoints, same evaluation examples).

Reported per (method, horizon):
    mean / IQM of the per-group mean report-split loss gain, with 95% bootstrap CIs
    paired difference vs. a reference method, with CI and P(group improves)
    success, acceptance, validity and clipping rates
    median step-equivalence and cost in fb-pass equivalents
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

STAGES = ("early", "mid", "late")


def iqm(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n == 0:
        return float("nan")
    lo, hi = int(np.floor(0.25 * n)), int(np.ceil(0.75 * n))
    mid = x[lo:hi] if hi > lo else x
    return float(mid.mean())


def bootstrap_ci(values: np.ndarray, stat=np.mean, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05
                 ) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(len(values), size=(n_boot, len(values)))
    stats = np.array([stat(values[i]) for i in idx])
    return float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2))


def _gain(row: dict, gated: bool, key: str = "loss") -> float | None:
    if gated:
        return row["gated_gains"]["report"][key]
    return row["gains"]["report"][key] if row["valid"] else None


def _group_means(rows: list[dict], fn) -> dict[str, float]:
    acc: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        v = fn(r)
        if v is not None:
            acc[r["group_id"]].append(v)
    return {g: float(np.mean(v)) for g, v in acc.items() if v}


def summarize(rows: list[dict], reference: str = "no_update", n_boot: int = 2000, seed: int = 0,
              stage: str | None = None) -> list[dict]:
    rows = [r for r in rows if stage is None or r["stage"] == stage]
    by_key: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for r in rows:
        if "requested_class" not in r:  # class-request rows are analysed by class_specificity / steering
            by_key[(r["method"], r["horizon"])].append(r)
    ref_index = {(r["source_id"], r["horizon"]): r for r in rows if r["method"] == reference}
    out = []
    for (method, h), rs in sorted(by_key.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        g_ungated = _group_means(rs, lambda r: _gain(r, False))
        g_gated = _group_means(rs, lambda r: _gain(r, True))
        g_acc = _group_means(rs, lambda r: _gain(r, False, "acc"))
        diffs = _group_means(
            rs,
            lambda r: (_gain(r, False) - _gain(ref_index[(r["source_id"], h)], False))
            if (r["source_id"], h) in ref_index and r["valid"] and ref_index[(r["source_id"], h)]["valid"] else None,
        )
        gdiffs = _group_means(
            rs,
            lambda r: (_gain(r, True) - _gain(ref_index[(r["source_id"], h)], True))
            if (r["source_id"], h) in ref_index else None,
        )
        vals = np.array(list(g_ungated.values()))
        dvals = np.array(list(diffs.values()))
        gdvals = np.array(list(gdiffs.values()))
        eq = [r["equivalent_steps"] if r["equivalent_steps"] is not None else np.inf for r in rs
              if r["valid"] and r.get("reference_max_steps") is not None]
        valid = [r for r in rs if r["valid"]]
        out.append({
            "method": method,
            "horizon": h,
            "stage": stage or "all",
            "n_rows": len(rs),
            "n_groups": len(g_ungated),
            "gain_mean": float(vals.mean()) if len(vals) else float("nan"),
            "gain_ci": bootstrap_ci(vals, np.mean, n_boot, seed),
            "gain_iqm": iqm(vals),
            "gated_gain_mean": float(np.mean(list(g_gated.values()))) if g_gated else float("nan"),
            "acc_gain_mean": float(np.mean(list(g_acc.values()))) if g_acc else float("nan"),
            "diff_vs_ref_mean": float(dvals.mean()) if len(dvals) else float("nan"),
            "diff_vs_ref_ci": bootstrap_ci(dvals, np.mean, n_boot, seed),
            "p_group_improves_vs_ref": float((dvals > 0).mean()) if len(dvals) else float("nan"),
            "gated_diff_vs_ref_mean": float(gdvals.mean()) if len(gdvals) else float("nan"),
            "gated_diff_vs_ref_ci": bootstrap_ci(gdvals, np.mean, n_boot, seed),
            "p_group_gated_improves_vs_ref": float((gdvals > 0).mean()) if len(gdvals) else float("nan"),
            "success_rate": float(np.mean([_gain(r, False) > 0 for r in valid])) if valid else float("nan"),
            "accept_rate": float(np.mean([r["accepted"] for r in rs])),
            "valid_rate": float(np.mean([r["valid"] for r in rs])),
            "clip_rate": float(np.mean([bool(r["safety"].get("clipped")) for r in rs])),
            "equiv_steps_median": float(np.median(eq)) if eq else float("nan"),
            "beyond_reference_rate": float(np.mean([np.isinf(e) for e in eq])) if eq else float("nan"),
            "gap_closed_median": float(np.median([r["gap_closed"] for r in valid if r["gap_closed"] is not None]))
            if any(r["gap_closed"] is not None for r in valid) else float("nan"),
            "cos_to_adamw_full_mean": float(np.mean([r["cos_to_adamw_full"] for r in valid
                                                     if r["cos_to_adamw_full"] is not None]))
            if any(r["cos_to_adamw_full"] is not None for r in valid) else float("nan"),
            "fb_equivalent_mean": float(np.mean([r["cost"]["fb_equivalent"] for r in rs])),
        })
    return out


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def controllability(rows: list[dict], n_boot: int = 2000, seed: int = 0) -> dict:
    """Does the achieved gain track the requested gain? (operator_q* request sweep)

    Per (source, horizon): Spearman between requested and achieved report-split
    loss gain across the sweep, averaged per group, bootstrapped over groups.
    """
    sweep = [r for r in rows if r["method"].startswith("operator_q") and r["valid"]]
    per_case: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for r in sweep:
        per_case[(r["source_id"], r["horizon"])].append(r)
    by_group: dict[str, list[float]] = defaultdict(list)
    for (_, _), rs in per_case.items():
        req = np.array([r["condition"]["loss"] for r in rs])
        got = np.array([r["gains"]["report"]["loss"] for r in rs])
        rho = _spearman(req, got)
        if not np.isnan(rho):
            by_group[rs[0]["group_id"]].append(rho)
    vals = np.array([np.mean(v) for v in by_group.values()])
    return {
        "n_cases": len(per_case),
        "n_groups": len(vals),
        "spearman_mean": float(vals.mean()) if len(vals) else float("nan"),
        "spearman_ci": bootstrap_ci(vals, np.mean, n_boot, seed),
    }


def class_specificity(rows: list[dict], num_classes: int, n_boot: int = 2000, seed: int = 0,
                      method: str = "operator_class_request") -> dict:
    """Requested-class x achieved-class gain matrix (H3).

    ``matrix[r][k]`` is the mean report-split class-k loss gain when class r was
    requested. Specificity = mean diagonal minus mean off-diagonal, computed
    per group and bootstrapped over groups. Requests for other classes serve
    as the control for each class.
    """
    rs = [r for r in rows if r["method"] == method and r["valid"]]
    if not rs:
        return {"n": 0}
    mat = np.zeros((num_classes, num_classes))
    cnt = np.zeros(num_classes)
    by_group: dict[str, list[float]] = defaultdict(list)
    for r in rs:
        k = r["requested_class"]
        g = np.array([r["gains"]["report"][f"class_{j}"] for j in range(num_classes)])
        mat[k] += g
        cnt[k] += 1
        by_group[r["group_id"]].append(float(g[k] - np.delete(g, k).mean()))
    mat = mat / np.maximum(cnt, 1)[:, None]
    vals = np.array([np.mean(v) for v in by_group.values()])
    return {
        "n": len(rs),
        "n_groups": len(vals),
        "matrix": mat.tolist(),
        "diag_mean": float(np.mean(np.diag(mat))),
        "offdiag_mean": float((mat.sum() - np.trace(mat)) / (num_classes * (num_classes - 1))),
        "specificity_mean": float(vals.mean()),
        "specificity_ci": bootstrap_ci(vals, np.mean, n_boot, seed),
    }


def _fmt(x: float, nd: int = 4) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    if isinstance(x, float) and np.isinf(x):
        return "∞"
    return f"{x:.{nd}f}"


def _ci(ci: tuple[float, float], nd: int = 4) -> str:
    return "–" if any(np.isnan(c) for c in ci) else f"[{ci[0]:.{nd}f}, {ci[1]:.{nd}f}]"


def markdown_report(summary: list[dict], ctrl: dict, spec: dict, reference: str, title: str) -> str:
    lines = [f"# {title}", "",
             f"Unit of analysis: init group. Gains are report-split NLL improvements (positive = better), "
             f"ungated unless noted. Paired differences are vs `{reference}`.", ""]
    for h in sorted({s["horizon"] for s in summary}):
        lines += [f"## Horizon {h} steps", "",
                  "| method | groups | gain mean | 95% CI | IQM | gated gain | Δacc | diff vs ref | diff CI | "
                  "P(group>ref) | success | accept | clip | equiv steps (median) | beyond ref | gap closed | "
                  "cos(AdamW) | cost (fb eq) |",
                  "|" + "---|" * 18]
        for s in sorted((s for s in summary if s["horizon"] == h), key=lambda s: -np.nan_to_num(s["gain_mean"], nan=-9)):
            lines.append(
                f"| {s['method']} | {s['n_groups']} | {_fmt(s['gain_mean'])} | {_ci(s['gain_ci'])} | "
                f"{_fmt(s['gain_iqm'])} | {_fmt(s['gated_gain_mean'])} | {_fmt(s['acc_gain_mean'])} | "
                f"{_fmt(s['diff_vs_ref_mean'])} | {_ci(s['diff_vs_ref_ci'])} | {_fmt(s['p_group_improves_vs_ref'], 2)} | "
                f"{_fmt(s['success_rate'], 2)} | {_fmt(s['accept_rate'], 2)} | {_fmt(s['clip_rate'], 2)} | "
                f"{_fmt(s['equiv_steps_median'], 1)} | {_fmt(s['beyond_reference_rate'], 2)} | "
                f"{_fmt(s['gap_closed_median'], 3)} | {_fmt(s['cos_to_adamw_full_mean'], 3)} | "
                f"{_fmt(s['fb_equivalent_mean'], 1)} |")
        lines.append("")
    lines += ["## Controllability (requested vs achieved loss gain, operator request sweep)", "",
              f"Spearman mean {_fmt(ctrl.get('spearman_mean'), 3)} CI {_ci(ctrl.get('spearman_ci', (np.nan, np.nan)), 3)} "
              f"over {ctrl.get('n_groups', 0)} groups / {ctrl.get('n_cases', 0)} cases.", ""]
    if spec.get("n"):
        lines += ["## Class-conditional specificity (H3)", "",
                  f"Diagonal (requested class) mean gain {_fmt(spec['diag_mean'])}, off-diagonal {_fmt(spec['offdiag_mean'])}; "
                  f"per-group specificity {_fmt(spec['specificity_mean'])} CI {_ci(spec['specificity_ci'])} "
                  f"over {spec['n_groups']} groups.", ""]
    return "\n".join(lines)


def selector_choices_table(rows: list[dict]) -> str:
    """How often each selector picked each candidate, per horizon and training stage."""
    counts: dict[tuple, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in rows:
        if r["method"].startswith("select_") and "chosen" in r.get("info", {}):
            counts[(r["method"], r["horizon"], r["stage"])][r["info"]["chosen"]] += 1
    if not counts:
        return ""
    lines = ["Selector choices (count of parents):", "", "| selector | horizon | stage | choices |", "|---|---|---|---|"]
    for (m, h, st), c in sorted(counts.items(), key=lambda kv: (kv[0][0], kv[0][1], STAGES.index(kv[0][2]))):
        lines.append(f"| {m} | {h} | {st} | " + ", ".join(f"{k}: {v}" for k, v in sorted(c.items())) + " |")
    return "\n".join(lines) + "\n"


def generalization_gap(train_rows: list[dict], heldout_rows: list[dict], n_boot: int = 2000, seed: int = 0,
                       gated: bool = False) -> list[dict]:
    """Train-root vs held-out-root gain per (method, horizon) -- the memorization measurement (H2).

    Groups differ between the two sets, so the comparison is unpaired: a two-sample bootstrap
    over groups of the difference in means. Baselines that never saw any root estimate how much
    the two root sets differ by chance; the operator's gap is read against theirs.
    """
    def per_group(rows: list[dict]) -> dict[tuple[str, int], np.ndarray]:
        out = {}
        keys = {(r["method"], r["horizon"]) for r in rows if r["method"] != "operator_class_request"}
        for m, h in keys:
            g = _group_means([r for r in rows if r["method"] == m and r["horizon"] == h],
                             lambda r: _gain(r, gated))
            out[(m, h)] = np.array(list(g.values()))
        return out

    tr, ho = per_group(train_rows), per_group(heldout_rows)
    rng = np.random.default_rng(seed)
    out = []
    for key in sorted(set(tr) & set(ho), key=lambda k: (k[1], k[0])):
        a, b = tr[key], ho[key]
        if len(a) < 2 or len(b) < 2:
            continue
        boots = [a[rng.integers(len(a), size=len(a))].mean() - b[rng.integers(len(b), size=len(b))].mean()
                 for _ in range(n_boot)]
        out.append({
            "method": key[0], "horizon": key[1], "gated": gated,
            "n_train_groups": len(a), "n_heldout_groups": len(b),
            "train_mean": float(a.mean()), "heldout_mean": float(b.mean()),
            "gap": float(a.mean() - b.mean()),
            "gap_ci": (float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))),
        })
    return out
