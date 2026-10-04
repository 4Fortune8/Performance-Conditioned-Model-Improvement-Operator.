"""Gate 1: is the operator's class steering better than class-weighted fine-tuning at equal compute?

Pre-registered in docs/EXPERIMENTS.md §6. Unit: (validation source, requested class). Per unit and
method:

    spec       report-split class-k loss gain minus the mean of the other classes' gains (ungated)
    spec_acc   the same with class accuracy
    target     report-split class-k loss gain
    overall    report-split overall NLL gain (the side effect of steering)
    spec_accept  ``spec`` on the accept split (used only to pick the baseline's class weight)

The operator's unit values are averaged over its request horizons. Units are averaged within each
init group; differences are paired on the same units and bootstrapped over groups.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from mio.config import parse_class_baseline
from mio.evaluation.analysis import _ci, _fmt, bootstrap_ci

METRICS = ("spec", "spec_acc", "target", "overall", "spec_accept")


def _unit_metrics(r: dict, num_classes: int) -> dict[str, float]:
    k = r["requested_class"]

    def spec(vals):
        vals = np.asarray(vals, dtype=float)
        return float(vals[k] - np.delete(vals, k).mean())

    g, ga = r["gains"]["report"], r["gains"]["accept"]
    acc_gain = (np.asarray(r["child_metrics"]["report"]["class_acc"])
                - np.asarray(r["parent_metrics"]["report"]["class_acc"]))
    return {
        "spec": spec([g[f"class_{j}"] for j in range(num_classes)]),
        "spec_acc": spec(acc_gain),
        "target": g[f"class_{k}"],
        "overall": g["loss"],
        "spec_accept": spec([ga[f"class_{j}"] for j in range(num_classes)]),
    }


def unit_table(rows: list[dict], method: str, num_classes: int, horizon: int | None = None
               ) -> dict[tuple[str, int], dict]:
    """(source_id, class) -> metrics averaged over horizons (and the group id)."""
    acc: dict[tuple[str, int], list[dict]] = defaultdict(list)
    group: dict[tuple[str, int], str] = {}
    for r in rows:
        if r["method"] != method or not r["valid"] or (horizon is not None and r["horizon"] != horizon):
            continue
        key = (r["source_id"], r["requested_class"])
        acc[key].append(_unit_metrics(r, num_classes))
        group[key] = r["group_id"]
    return {key: {"group_id": group[key], **{m: float(np.mean([v[m] for v in vs])) for m in METRICS}}
            for key, vs in acc.items()}


def _by_group(units: dict, fn) -> np.ndarray:
    acc: dict[str, list[float]] = defaultdict(list)
    for key, u in units.items():
        v = fn(key, u)
        if v is not None:
            acc[u["group_id"]].append(v)
    return np.array([np.mean(v) for _, v in sorted(acc.items())])


def describe(units: dict, n_boot: int, seed: int) -> dict:
    out = {"n_units": len(units)}
    for m in METRICS:
        vals = _by_group(units, lambda k, u: u[m])
        out[m] = float(vals.mean()) if len(vals) else float("nan")
        out[f"{m}_ci"] = bootstrap_ci(vals, np.mean, n_boot, seed)
    out["n_groups"] = len(_by_group(units, lambda k, u: 0.0))
    return out


def paired(a: dict, b: dict, metric: str, n_boot: int, seed: int) -> dict:
    vals = _by_group(a, lambda k, u: u[metric] - b[k][metric] if k in b else None)
    return {"mean": float(vals.mean()) if len(vals) else float("nan"),
            "ci": bootstrap_ci(vals, np.mean, n_boot, seed),
            "p_group_positive": float((vals > 0).mean()) if len(vals) else float("nan"),
            "n_groups": len(vals)}


def _equivalent_steps(op_spec: float, ladder: list[tuple[int, float]]) -> str:
    """Fine-tuning steps whose specificity equals the operator's (log-step interpolation)."""
    ladder = sorted(ladder)
    if op_spec <= ladder[0][1]:
        return f"< {ladder[0][0]}"
    for (n0, s0), (n1, s1) in zip(ladder, ladder[1:]):
        if s0 < op_spec <= s1:
            f = (op_spec - s0) / (s1 - s0)
            return f"{math.exp(math.log(n0) + f * (math.log(n1) - math.log(n0))):.0f}"
    return f"> {ladder[-1][0]} (not monotone or not reached)"


def gate1(rows: list[dict], num_classes: int, matched_steps: int, operator: str = "operator_class_request",
          n_boot: int = 2000, seed: int = 0) -> dict:
    op = unit_table(rows, operator, num_classes)
    names = sorted({r["method"] for r in rows if r["method"].startswith("class_ft_")})
    base = {n: unit_table(rows, n, num_classes) for n in names}
    budgets: dict[int, list[str]] = defaultdict(list)
    for n in names:
        budgets[parse_class_baseline(n)[1]].append(n)

    out: dict = {"operator": describe(op, n_boot, seed), "matched_steps": matched_steps,
                 "operator_by_horizon": {}, "baselines": {}, "bar": {}}
    for h in sorted({r["horizon"] for r in rows if r["method"] == operator}):
        out["operator_by_horizon"][h] = describe(unit_table(rows, operator, num_classes, h), n_boot, seed)
    for n, units in base.items():
        out["baselines"][n] = {**describe(units, n_boot, seed),
                               "spec_vs_operator": paired(op, units, "spec", n_boot, seed)}
    for mult in sorted(budgets):
        # class weight selected on the accept split only (pooled mean accept specificity)
        chosen = max(budgets[mult], key=lambda n: np.mean([u["spec_accept"] for u in base[n].values()]))
        b = base[chosen]
        out["bar"][mult] = {
            "chosen": chosen,
            "steps": mult * matched_steps,
            "accept_spec": {n: float(np.mean([u["spec_accept"] for u in base[n].values()])) for n in budgets[mult]},
            **{f"diff_{m}": paired(op, b, m, n_boot, seed) for m in ("spec", "spec_acc", "target", "overall")},
            "bar_spec": out["baselines"][chosen]["spec"],
        }
    if out["bar"]:
        ladder = [(v["steps"], v["bar_spec"]) for v in out["bar"].values()]
        out["equivalent_ft_steps"] = _equivalent_steps(out["operator"]["spec"], ladder)
    if 1 in out["bar"]:
        d, side = out["bar"][1]["diff_spec"], out["bar"][1]["diff_overall"]
        if d["ci"][0] > 0 and side["ci"][1] >= 0:
            decision = "GO"
        elif d["ci"][1] < 0:
            decision = "NO-GO"
        else:
            decision = "INCONCLUSIVE"
        out["decision"] = decision
    return out


def gate1_markdown(res: dict, title: str = "Gate 1: steering vs class-weighted fine-tuning") -> str:
    o = res["operator"]
    lines = [f"# {title}", "",
             "Validation roots × dev split. Unit = (source, requested class); per-group means, "
             "bootstrap CIs over groups; differences paired on the same units (operator − baseline).", "",
             f"**Decision (pre-registered rule, EXPERIMENTS.md §6): {res.get('decision', 'n/a')}**", "",
             f"Matched budget: {res['matched_steps']} AdamW steps (= operator cost in fb-pass equivalents). "
             f"Equivalent fine-tuning steps for the operator's specificity: {res.get('equivalent_ft_steps', '–')}.",
             "", "## Arms", "",
             "| method | steps | specificity [CI] | class-acc spec. | requested-class gain | overall NLL gain | "
             "accept spec. | Δspec vs operator [CI] |",
             "|---|---|---|---|---|---|---|---|"]

    def arm(name, steps, d, diff=None):
        dd = (f"{_fmt(-diff['mean'])} {_ci((-diff['ci'][1], -diff['ci'][0]))}" if diff else "—")
        return (f"| {name} | {steps} | {_fmt(d['spec'])} {_ci(d['spec_ci'])} | {_fmt(d['spec_acc'])} | "
                f"{_fmt(d['target'])} | {_fmt(d['overall'])} | {_fmt(d['spec_accept'])} | {dd} |")

    lines.append(arm("operator (conditioned)", f"{res['matched_steps']} (fb-eq)", o))
    for h, d in res["operator_by_horizon"].items():
        lines.append(arm(f"  operator, h={h} request", "", d))
    for n, d in sorted(res["baselines"].items(), key=lambda kv: (parse_class_baseline(kv[0])[1],
                                                                  parse_class_baseline(kv[0])[0])):
        w, mult = parse_class_baseline(n)
        lines.append(arm(n, mult * res["matched_steps"], d, d["spec_vs_operator"]))
    lines += ["", "(Δspec column: baseline minus operator.)", "",
              "## Operator vs the bar at each budget (class weight chosen on the accept split)", "",
              "| budget | bar | Δ specificity [CI] (groups op > bar) | Δ class-acc spec. [CI] | "
              "Δ requested-class gain [CI] | Δ overall NLL gain [CI] |",
              "|---|---|---|---|---|---|"]
    for mult, b in res["bar"].items():
        cells = []
        for m in ("spec", "spec_acc", "target", "overall"):
            d = b[f"diff_{m}"]
            extra = f" ({d['p_group_positive'] * d['n_groups']:.0f}/{d['n_groups']})" if m == "spec" else ""
            cells.append(f"{_fmt(d['mean'])} {_ci(d['ci'])}{extra}")
        lines.append(f"| {mult}× ({b['steps']} steps) | {b['chosen']} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
