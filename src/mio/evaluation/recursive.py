"""Repeated operator application with stopping conditions (H5, extension).

After each accepted step the operator's own move becomes the "recent
trajectory": history at lag L is synthesized as ``theta - (L / h) * delta``
(uniform motion over the step) and the optimizer state is carried over
unchanged. This is exactly the input distribution shift discussed in
docs/DESIGN_REVIEW.md; interleaving real optimizer steps is the alternative.
Every proposal, accepted or rejected, is logged; rejection rolls back.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from mio.evaluation.evaluator import ChildEngine
from mio.state import Condition, ModelState


@dataclass
class RecursionLimits:
    max_iters: int = 5
    max_cum_rel_update: float = 1.0  # sum of ||delta|| / ||theta_0||
    min_gain: float = 0.0  # accept-split loss gain required to continue
    max_consecutive_rejections: int = 1


@dataclass
class RecursionResult:
    final_state: ModelState
    steps: list[dict] = field(default_factory=list)
    stop_reason: str = ""


def recursive_improve(improver, state: ModelState, engine: ChildEngine, condition_fn, limits: RecursionLimits,
                      parent_metrics: dict) -> RecursionResult:
    """``condition_fn(state) -> Condition`` chooses the request at each iteration."""
    theta0_norm = float(state.theta.norm())
    cum = 0.0
    rejections = 0
    result = RecursionResult(final_state=state)
    current, current_metrics = state, parent_metrics
    for it in range(limits.max_iters):
        cond: Condition = condition_fn(current)
        proposal = improver.propose(current, cond)
        rec = engine.run(current, proposal, current_metrics)
        delta = rec.pop("_delta", None)
        ok = rec["valid"] and rec["gains"]["accept"]["loss"] > limits.min_gain
        step_rel = float(delta.norm()) / theta0_norm if delta is not None else 0.0
        result.steps.append({"iter": it, "horizon": cond.horizon, "condition": cond.gains, "valid": rec["valid"],
                             "accepted": ok, "gains": rec.get("gains"), "safety": rec["safety"],
                             "delta_rel_to_theta0": step_rel})
        if not rec["valid"]:
            result.stop_reason = f"invalid proposal ({rec['safety'].get('rejected')})"
            break
        if not ok:
            rejections += 1
            if rejections >= limits.max_consecutive_rejections:
                result.stop_reason = "no further measured improvement"
                break
            continue
        if cum + step_rel > limits.max_cum_rel_update:
            result.steps[-1]["accepted"] = False
            result.stop_reason = "cumulative update limit"
            break
        rejections = 0
        cum += step_rel
        new_theta = current.theta + delta
        history = {lag: new_theta - (lag / cond.horizon) * delta for lag in current.history}
        current = replace(current, theta=new_theta, step=current.step + cond.horizon, history=history,
                          metrics=rec["child_metrics"]["accept"], checkpoint_id=f"{current.checkpoint_id}+op{it}")
        current_metrics = rec["child_metrics"]
    else:
        result.stop_reason = "max iterations"
    result.final_state = current
    return result
