"""Global weight-rescaling baseline: delta = alpha * theta.

Scaling every weight and bias of a ReLU network by (1 + alpha) rescales the
logits, so on held-out data this acts like a confidence (temperature)
adjustment, and for alpha < 0 like weight-decay shrinkage. A learned operator
whose gains match a tuned rescaling has learned calibration, not optimization.
"""

from __future__ import annotations

from mio.baselines.extrapolation import _alpha
from mio.state import Condition, ModelState, Proposal


class WeightScaling:
    name = "weight_scaling"

    def __init__(self, alpha: float | dict[int, float] = 0.0, name: str = "weight_scaling"):
        self.alpha = alpha
        self.name = name

    def propose(self, state: ModelState, condition: Condition) -> Proposal:
        a = _alpha(self.alpha, condition.horizon)
        return Proposal(delta=a * state.theta, info={"alpha": a})
