"""Interleaved application protocol (E4 / H4): optimizer for k steps -> jump -> repeat.

This is the regime where weight-nowcasting methods (Introspection, WNN, NiNo)
report their gains. Starting from a stored anchor, each cycle runs ``k``
AdamW steps (recording the lag snapshots the operator needs), then lets an
improver propose a jump of ``horizon`` step-equivalents, gated on the accept
split. The plain-AdamW reference uses the *same* data order, so the
comparison is paired. Reported per (source, method):

    gain_vs_reference   reference loss - interleaved loss at equal AdamW steps (report split)
    speedup             reference steps needed to match the final interleaved loss
                        / AdamW steps actually used, against the reference's best-so-far
                        envelope (AdamW with best-checkpoint selection; None = beyond reference)

``no_update`` reproduces the reference exactly (gain 0; speedup 1 while the reference is
still improving, < 1 once it overfits), which is a calibration check.
"""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from mio.baselines.conventional import equivalent_steps, settings_from_state
from mio.config import ExperimentConfig
from mio.datasets.tasks import load_task
from mio.datasets.transitions import load_transitions
from mio.evaluation.analysis import bootstrap_ci
from mio.evaluation.evaluator import ChildEngine, build_methods
from mio.evaluation.metrics import evaluate
from mio.models.registry import build_model
from mio.state import ModelState
from mio.trajectories.checkpoints import CheckpointStore
from mio.trajectories.training import EMATracker, train_steps
from mio.utils.logging import get_logger
from mio.utils.reproducibility import configure_determinism, derive_seed, git_commit
from mio.utils.serialization import write_json, write_jsonl

log = get_logger("mio.interleaved")


def interleaved_run(model, task, engine: ChildEngine, state0: ModelState, improver, condition_fn, k: int,
                    cycles: int, horizon: int, lags: list[int], betas_eps: tuple[float, float, float],
                    order_seed: int, gate: bool = True) -> list[dict]:
    """Returns the report-split loss after each cycle (``improver=None`` = plain AdamW)."""
    if k < max(lags):
        raise ValueError(f"adam steps per cycle ({k}) must cover the largest history lag ({max(lags)})")
    settings = settings_from_state(state0, order_seed, *betas_eps)
    train = task.split("train")
    theta, opt, step, data_step = state0.theta.clone(), state0.opt.clone(), state0.step, 0
    report = engine.report_split

    def loss(th):
        return evaluate(model, th, report, task.num_classes)["loss"]

    ema = EMATracker.from_corrected(state0.ema, step)  # continue the parent's EMAs through the run
    points = [{"adam_steps": 0, "loss": loss(theta), "jump": None}]
    for c in range(cycles):
        snaps: dict[int, object] = {}
        need = {k - lag: lag for lag in lags}

        def cb(j, th, _opt, _need=need, _snaps=snaps):
            ema.update(th)
            if j in _need:
                _snaps[_need[j]] = th.clone()

        res = train_steps(model, theta, opt, train, settings, k, data_step_start=data_step, callback=cb)
        theta, opt = res.theta, res.opt
        data_step += k
        step += k
        jump = None
        if improver is not None:
            m = engine.measure(theta)
            state = ModelState(theta=theta, step=step, history=snaps, opt=opt, ema=ema.corrected(),
                               metrics=m["accept"],
                               hparams=state0.hparams, checkpoint_id=f"{state0.checkpoint_id}+cycle{c}",
                               root_id=state0.root_id, group_id=state0.group_id)
            rec = engine.run(state, improver.propose(state, condition_fn(state, horizon)), m)
            delta = rec.pop("_delta", None)
            take = rec["valid"] and (rec["accepted"] or not gate)
            if take:
                theta = theta + delta
            jump = {"valid": rec["valid"], "accepted": rec["accepted"], "taken": take,
                    "accept_gain": rec["gains"]["accept"]["loss"] if rec["valid"] else None}
        points.append({"adam_steps": data_step, "loss": loss(theta), "jump": jump})
    return points


def run_interleaved(cfg: ExperimentConfig, root_split: str | None = None, methods: list[str] | None = None,
                    out_dir: Path | None = None) -> Path:
    ec, ic = cfg.evaluation, cfg.evaluation.interleaved
    root_split = root_split or ec.root_split
    if root_split == "test" or ec.report_split == "test":
        raise RuntimeError("interleaved runs are development experiments; use val roots and the dev split")
    methods = list(methods or ic.methods)
    configure_determinism(cfg.num_threads)
    t0 = time.perf_counter()
    store = CheckpointStore(cfg.population_dir)
    task = load_task(cfg.task, cfg.data_dir)
    model = build_model(cfg.model, task.input_dim, task.num_classes)
    store.spec.check_compatible(model.spec)
    engine = ChildEngine(model, task, cfg.safety, ec.report_split, False, ec.accept_min_gain)
    train_tr = [t for t in load_transitions(cfg.transitions_path) if t["partition"] == "train"]
    improvers, operator, policy, alphas, _ = build_methods(
        cfg, store, task, model, engine, [m for m in methods if m != "no_update"], train_tr)
    oc = cfg.population.optimizer
    lags = store.meta["history_lags"]
    bse = (oc.beta1, oc.beta2, oc.eps)

    def condition_fn(state, h):
        return policy.request(state, h, "loss", ec.request_quantile, ec.request_neighbors)

    sources = [r for r in store.sources(root_split) if r["step"] in set(ic.start_steps)]
    rows, ref_curves = [], []
    for src in sources:
        state0 = store.model_state(src["id"])
        seed = derive_seed(ec.seed, "interleaved", src["id"])
        ref = interleaved_run(model, task, engine, state0, None, condition_fn, ic.adam_steps,
                              ic.cycles * ic.reference_multiple, ic.horizon, lags, bse, seed)
        ref_curve = [(p["adam_steps"], p["loss"]) for p in ref]
        ref_curves.append({"source_id": src["id"], "group_id": src["group_id"], "curve": ref_curve})
        ref_at = {p["adam_steps"]: p["loss"] for p in ref}
        runs = [("no_update", None)] + [(imp.name, imp) for imp in improvers]
        for name, imp in runs:
            pts = ref[: ic.cycles + 1] if imp is None else interleaved_run(
                model, task, engine, state0, imp, condition_fn, ic.adam_steps, ic.cycles, ic.horizon, lags, bse,
                seed, ic.gate)
            final = pts[-1]
            eq = equivalent_steps(ref_curve, final["loss"])
            rows.append({
                "method": name, "source_id": src["id"], "root_id": src["root_id"], "group_id": src["group_id"],
                "start_step": src["step"], "adam_steps": final["adam_steps"],
                "final_loss": final["loss"], "reference_loss": ref_at[final["adam_steps"]],
                "gain_vs_reference": ref_at[final["adam_steps"]] - final["loss"],
                "speedup": (eq / final["adam_steps"]) if eq is not None else None,
                "jumps_taken": sum(1 for p in pts if p["jump"] and p["jump"]["taken"]),
                "curve": [(p["adam_steps"], p["loss"]) for p in pts],
            })
        log.info("  %s done", src["id"])

    out_dir = out_dir or cfg.results_dir / f"interleaved_{root_split}_{ec.report_split}"
    write_jsonl(out_dir / "results.jsonl", rows)
    write_jsonl(out_dir / "reference_curves.jsonl", ref_curves)
    summary = summarize_interleaved(rows, ec.bootstrap_samples, ec.seed)
    write_json(out_dir / "summary.json", {"summary": summary, "alphas": {k: {str(h): a for h, a in v.items()}
                                                                         for k, v in alphas.items()},
                                          "config": cfg.to_dict(), "git_commit": git_commit(),
                                          "wall_time_s": time.perf_counter() - t0})
    md = [f"# {cfg.name}: interleaved application ({root_split} roots, {ec.report_split} split)", "",
          f"{ic.cycles} cycles of {ic.adam_steps} AdamW steps + one jump of horizon {ic.horizon} "
          f"(gated: {ic.gate}); start steps {ic.start_steps}. Unit: init group.", "",
          "| method | groups | gain vs reference (mean) | 95% CI | P(group>0) | speedup (median) | beyond ref | jumps taken |",
          "|---|---|---|---|---|---|---|---|"]
    for s in summary:
        md.append(f"| {s['method']} | {s['n_groups']} | {s['gain_mean']:.4f} | [{s['gain_ci'][0]:.4f}, {s['gain_ci'][1]:.4f}] "
                  f"| {s['p_group_positive']:.2f} | {s['speedup_median']:.2f} | {s['beyond_rate']:.2f} | "
                  f"{s['jumps_taken_mean']:.1f} |")
    (out_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log.info("wrote %d interleaved rows to %s (%.1fs)", len(rows), out_dir, time.perf_counter() - t0)
    return out_dir


def summarize_interleaved(rows: list[dict], n_boot: int = 2000, seed: int = 0) -> list[dict]:
    by_method: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_method[r["method"]].append(r)
    out = []
    for method, rs in by_method.items():
        per_group: dict[str, list[float]] = defaultdict(list)
        for r in rs:
            per_group[r["group_id"]].append(r["gain_vs_reference"])
        g = np.array([np.mean(v) for v in per_group.values()])
        sp = [r["speedup"] if r["speedup"] is not None else np.inf for r in rs]
        out.append({
            "method": method,
            "n_groups": len(g),
            "gain_mean": float(g.mean()),
            "gain_ci": bootstrap_ci(g, np.mean, n_boot, seed),
            "p_group_positive": float((g > 0).mean()),
            "speedup_median": float(np.median(sp)),
            "beyond_rate": float(np.mean([np.isinf(x) for x in sp])),
            "jumps_taken_mean": float(np.mean([r["jumps_taken"] for r in rs])),
        })
    return sorted(out, key=lambda s: -s["gain_mean"])
