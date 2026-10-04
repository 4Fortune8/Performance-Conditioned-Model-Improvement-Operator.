"""Child-model engine and the comparative evaluation protocol.

``ChildEngine`` treats every improver identically:

1. structural check of the proposed delta (shape / ParamSpec), NaN/Inf check
2. per-tensor magnitude limit ||delta_T|| <= max_rel_update * ||theta_T||
3. build the child, check it is finite and bounded
4. measure the child on the *accept* split (gating) and the *report* split
5. accept iff the accept-split loss gain exceeds ``accept_min_gain``,
   otherwise roll back to the parent

Both the ungated outcome (what the method proposed) and the gated outcome
(after accept/rollback) are recorded, so gating is applied equally to all
methods and can also be ignored in analysis. Selection happens on ``accept``;
reported numbers come from a different split (``dev`` during development,
``test`` only in the final run), which avoids the winner's curse.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mio.baselines.averaging import HistoryAverage
from mio.baselines.controls import ForeignDelta, RandomNormMatched
from mio.baselines.conventional import ContinuedAdamW, NoUpdate, equivalent_steps, reference_curve
from mio.baselines.extrapolation import AdamExtrapolation, LinearExtrapolation, Scaled, tune_alpha
from mio.config import ExperimentConfig, SafetyConfig
from mio.datasets.tasks import TaskData, load_task
from mio.datasets.transitions import load_transitions
from mio.evaluation.metrics import evaluate, gains
from mio.models.registry import build_model
from mio.operators.residual import ConditionPolicy, LearnedOperator
from mio.state import Condition, ModelState, Proposal
from mio.trajectories.checkpoints import CheckpointStore
from mio.utils.logging import get_logger
from mio.utils.reproducibility import configure_determinism, derive_seed, git_commit
from mio.utils.serialization import write_json, write_jsonl

log = get_logger("mio.eval")


class ChildEngine:
    def __init__(self, model, task: TaskData, safety: SafetyConfig, report_split: str = "dev",
                 allow_test: bool = False, accept_min_gain: float = 0.0):
        self.model = model
        self.spec = model.spec
        self.num_classes = task.num_classes
        self.safety = safety
        self.accept_split = task.split("accept")
        self.report_split_name = report_split
        self.report_split = task.split(report_split, allow_test=allow_test)
        self.accept_min_gain = accept_min_gain

    def sanitize(self, theta: torch.Tensor, delta: torch.Tensor) -> tuple[torch.Tensor | None, dict]:
        self.spec.check_vector(delta, check_finite=False)  # raises IncompatibleModelError on structure mismatch
        if not torch.isfinite(delta).all():
            return None, {"rejected": "nonfinite_delta"}
        norms = self.spec.per_tensor_norm(delta)
        limits = self.safety.max_rel_update * self.spec.per_tensor_norm(theta)
        factor = torch.where(norms > limits, limits / norms.clamp_min(1e-30), torch.ones_like(norms))
        clipped = bool((factor < 1).any())
        if clipped:
            delta = delta * self.spec.expand(factor)
        return delta, {"clipped": clipped, "clip_min_factor": float(factor.min())}

    def measure(self, theta: torch.Tensor) -> dict:
        return {
            "accept": evaluate(self.model, theta, self.accept_split, self.num_classes),
            "report": evaluate(self.model, theta, self.report_split, self.num_classes),
        }

    def run(self, state: ModelState, proposal: Proposal, parent_metrics: dict) -> dict:
        delta, info = self.sanitize(state.theta, proposal.delta)
        out: dict = {"safety": info, "valid": False, "accepted": False}
        if delta is None:
            out["gated_gains"] = {"report": _zero_gains(parent_metrics["report"])}
            return out
        child = state.theta + delta
        if not torch.isfinite(child).all() or float(child.abs().max()) > self.safety.max_abs_value:
            out["safety"]["rejected"] = "invalid_child"
            out["gated_gains"] = {"report": _zero_gains(parent_metrics["report"])}
            return out
        m = self.measure(child)
        g = {s: gains(parent_metrics[s], m[s]) for s in ("accept", "report")}
        accepted = g["accept"]["loss"] > self.accept_min_gain
        out.update(
            valid=True,
            accepted=accepted,
            child_metrics=m,
            gains=g,
            gated_gains={"report": g["report"] if accepted else _zero_gains(parent_metrics["report"])},
            delta_norm=float(delta.norm()),
            delta_rel_norm=float(delta.norm() / state.theta.norm()),
        )
        out["_delta"] = delta
        return out


def _zero_gains(metrics: dict) -> dict:
    return {"loss": 0.0, "acc": 0.0, **{f"class_{k}": 0.0 for k in range(len(metrics["class_loss"]))}}


def _spread(items: list, n: int | None) -> list:
    if n is None or len(items) <= n:
        return items
    idx = np.unique(np.linspace(0, len(items) - 1, n).round().astype(int))
    return [items[i] for i in idx]


def stage_of(step: int, total_steps: int) -> str:
    frac = step / total_steps
    return "early" if frac < 0.1 else ("mid" if frac < 0.5 else "late")


def run_evaluation(cfg: ExperimentConfig, root_split: str | None = None, report_split: str | None = None,
                   allow_test: bool = False, methods: list[str] | None = None,
                   operator_path: Path | None = None, out_dir: Path | None = None) -> Path:
    ec = cfg.evaluation
    root_split = root_split or ec.root_split
    report_split = report_split or ec.report_split
    if (root_split == "test" or report_split == "test") and not allow_test:
        raise RuntimeError("evaluating on test roots / the test split requires allow_test=True (final run only)")
    methods = list(methods or ec.methods)
    configure_determinism(cfg.num_threads)
    t_start = time.perf_counter()

    store = CheckpointStore(cfg.population_dir)
    task = load_task(cfg.task, cfg.data_dir)
    model = build_model(cfg.model, task.input_dim, task.num_classes)
    store.spec.check_compatible(model.spec)
    train_split = task.split("train")
    oc = cfg.population.optimizer
    lags = store.meta["history_lags"]
    all_tr = load_transitions(cfg.transitions_path)
    train_tr = [t for t in all_tr if t["partition"] == "train"]

    engine = ChildEngine(model, task, cfg.safety, report_split, allow_test, ec.accept_min_gain)

    # ---- learned operator --------------------------------------------------
    operator = None
    if any(m.startswith("operator") for m in methods):
        operator_path = operator_path or cfg.results_dir / "operator" / "operator.pt"
        operator = LearnedOperator.load(operator_path, store.spec)
    policy = ConditionPolicy(train_tr, cfg.transitions.metric_split)

    # ---- tune baseline step sizes on validation roots (accept split only) ----
    tune_states = [store.model_state(r["id"]) for r in _spread(store.sources("val"), 24)]

    def accept_score(state: ModelState, proposal: Proposal) -> float:
        delta, _ = engine.sanitize(state.theta, proposal.delta)
        if delta is None:
            return -math.inf
        child = state.theta + delta
        if not torch.isfinite(child).all():
            return -math.inf
        return state.metrics["loss"] - evaluate(model, child, engine.accept_split, task.num_classes)["loss"]

    alphas = {}
    if "linear_extrapolation" in methods or "random_norm_matched" in methods:
        alphas["linear"] = tune_alpha(lambda a: LinearExtrapolation(max(lags), a), tune_states, ec.horizons,
                                      ec.alpha_grid, accept_score)
    if "adam_extrapolation" in methods:
        alphas["adam"] = tune_alpha(lambda a: AdamExtrapolation(oc.beta1, oc.beta2, oc.eps, a), tune_states,
                                    ec.horizons, ec.alpha_grid, accept_score)
    if operator is not None and "operator_scaled" in methods:
        alphas["operator"] = tune_alpha(
            lambda a: Scaled(operator, a), tune_states, ec.horizons, ec.alpha_grid, accept_score,
            condition_fn=lambda s, h: policy.request(s, h, "loss", ec.request_quantile, ec.request_neighbors))
    log.info("tuned alphas (val roots, accept split): %s", alphas)

    matched_steps = max(1, math.ceil(operator.flops() / (6.0 * store.spec.numel * oc.batch_size))) if operator else 1
    improvers = []
    for m in methods:
        if m == "no_update":
            improvers.append(NoUpdate())
        elif m == "linear_extrapolation":
            improvers.append(LinearExtrapolation(max(lags), alphas["linear"]))
        elif m == "adam_extrapolation":
            improvers.append(AdamExtrapolation(oc.beta1, oc.beta2, oc.eps, alphas["adam"]))
        elif m == "history_average":
            improvers.append(HistoryAverage())
        elif m == "random_norm_matched":
            improvers.append(RandomNormMatched(store.spec, LinearExtrapolation(max(lags), alphas["linear"]), ec.seed))
        elif m == "foreign_delta":
            improvers.append(ForeignDelta(store, train_tr))
        elif m == "adamw_matched":
            improvers.append(ContinuedAdamW(model, train_split, oc.beta1, oc.beta2, oc.eps, steps=matched_steps,
                                            seed=ec.seed, name="adamw_matched"))
        elif m == "adamw_full":
            improvers.append(ContinuedAdamW(model, train_split, oc.beta1, oc.beta2, oc.eps, steps=None, seed=ec.seed))
        elif m == "operator":
            improvers.append(operator)
        elif m == "operator_scaled":
            improvers.append(Scaled(operator, alphas["operator"], name="operator_scaled"))
        else:
            raise KeyError(f"unknown method {m!r}")
    # adamw_full first so other methods can be compared with the real optimizer's delta.
    improvers.sort(key=lambda imp: imp.name != "adamw_full")

    sources = []
    for rid in store.roots(root_split):
        rs = [r for r in store.sources(root_split) if r["root_id"] == rid]
        sources.extend(_spread(rs, ec.max_sources_per_root))
    log.info("evaluating %d sources from %d %s roots on report split %r with methods %s",
             len(sources), len(store.roots(root_split)), root_split, report_split, [i.name for i in improvers])

    best_loss: dict[str, float] = {}
    rows: list[dict] = []
    total_steps = cfg.population.total_steps
    for si, src in enumerate(sources):
        state = store.model_state(src["id"])
        parent = {"accept": state.metrics, "report": evaluate(model, state.theta, engine.report_split, task.num_classes)}
        if src["root_id"] not in best_loss:
            anchors = store.select(kind="anchor", line="trunk", root_id=src["root_id"])
            best_loss[src["root_id"]] = min(
                evaluate(model, store.theta(a["id"]), engine.report_split, task.num_classes)["loss"] for a in anchors
            )
        curve = reference_curve(
            model, state, train_split,
            lambda th: evaluate(model, th, engine.report_split, task.num_classes)["loss"],
            ec.reference_multiple * max(ec.horizons), ec.reference_eval_every, oc.beta1, oc.beta2, oc.eps,
            order_seed=derive_seed(ec.seed, "reference-curve", src["id"]),
        )
        for h in ec.horizons:
            base_cond = policy.request(state, h, "loss", ec.request_quantile, ec.request_neighbors)
            adam_delta = None
            jobs: list[tuple[str, object, Condition, dict]] = [(imp.name, imp, base_cond, {}) for imp in improvers]
            if operator is not None:
                for q in ec.request_sweep:
                    c = policy.request(state, h, "loss", q, ec.request_neighbors)
                    jobs.append((f"operator_q{q:g}", operator, c, {"request_quantile": q}))
                if ec.class_requests:
                    for k in range(task.num_classes):
                        c = policy.request(state, h, f"class_{k}", ec.request_quantile, ec.request_neighbors)
                        jobs.append(("operator_class_request", operator, c, {"requested_class": k}))
            for name, imp, cond, extra in jobs:
                proposal = imp.propose(state, cond)
                rec = engine.run(state, proposal, parent)
                delta = rec.pop("_delta", None)
                if name == "adamw_full" and delta is not None:
                    adam_delta = delta
                child_loss = rec["child_metrics"]["report"]["loss"] if rec["valid"] else None
                gap = best_loss[src["root_id"]]
                rows.append({
                    "method": name,
                    **extra,
                    "source_id": src["id"],
                    "root_id": src["root_id"],
                    "group_id": src["group_id"],
                    "partition": src["partition"],
                    "source_step": src["step"],
                    "stage": stage_of(src["step"], total_steps),
                    "horizon": h,
                    "condition": cond.gains,
                    "parent_metrics": parent,
                    "valid": rec["valid"],
                    "accepted": rec["accepted"],
                    "safety": rec["safety"],
                    "gains": rec.get("gains"),
                    "gated_gains": rec["gated_gains"],
                    "child_metrics": rec.get("child_metrics"),
                    "delta_norm": rec.get("delta_norm"),
                    "delta_rel_norm": rec.get("delta_rel_norm"),
                    "cos_to_adamw_full": (float(F.cosine_similarity(delta, adam_delta, dim=0))
                                          if delta is not None and adam_delta is not None else None),
                    "equivalent_steps": equivalent_steps(curve, child_loss) if child_loss is not None else 0.0,
                    "reference_max_steps": curve[-1][0],
                    "gap_closed": ((parent["report"]["loss"] - child_loss) / (parent["report"]["loss"] - gap)
                                   if child_loss is not None and parent["report"]["loss"] - gap > 1e-9 else None),
                    "cost": vars(proposal.cost),
                    "info": {k: v for k, v in proposal.info.items() if isinstance(v, (int, float, str))},
                })
        log.info("  [%d/%d] %s done", si + 1, len(sources), src["id"])

    out_dir = out_dir or cfg.results_dir / f"eval_{root_split}_{report_split}"
    write_jsonl(out_dir / "results.jsonl", rows)
    write_json(out_dir / "run_info.json", {
        "root_split": root_split,
        "report_split": report_split,
        "methods": [i.name for i in improvers],
        "alphas": {k: {str(h): a for h, a in v.items()} for k, v in alphas.items()},
        "matched_steps": matched_steps,
        "n_sources": len(sources),
        "n_roots": len(store.roots(root_split)),
        "operator_path": str(operator_path) if operator else None,
        "wall_time_s": time.perf_counter() - t_start,
        "git_commit": git_commit(),
        "config": cfg.to_dict(),
    })
    log.info("wrote %d result rows to %s (%.1fs)", len(rows), out_dir, time.perf_counter() - t_start)
    return out_dir
