"""Weight-averaging baselines over compatible checkpoints of the *same* trajectory."""

from __future__ import annotations

import torch

from mio.state import Condition, ModelState, Proposal


class HistoryAverage:
    """LAWA-style average of the current weights and the saved lag snapshots.

    Uses only checkpoints from the parent's own trajectory, so no alignment is
    needed. Costs no training compute; ignores the requested horizon.
    """

    name = "history_average"

    def propose(self, state: ModelState, condition: Condition) -> Proposal:
        if not state.history:
            raise ValueError("HistoryAverage needs history snapshots")
        stack = torch.stack([state.theta, *state.history.values()])
        return Proposal(delta=stack.mean(0) - state.theta, info={"n_averaged": len(stack)})


class EMAWeights:
    """Polyak/EMA averaging: move to the bias-corrected EMA of the trunk iterates.

    The EMA is tracked alongside training (``population.checkpoints.ema_decays``)
    at negligible cost and needs no alignment. Ignores the requested horizon.
    """

    def __init__(self, decay: float):
        self.decay = float(decay)
        self.name = f"ema_{self.decay:g}"

    def propose(self, state: ModelState, condition: Condition) -> Proposal:
        if self.decay not in state.ema:
            raise ValueError(f"state has no EMA with decay {self.decay:g} (have {sorted(state.ema)})")
        return Proposal(delta=state.ema[self.decay] - state.theta, info={"decay": self.decay})


def averaging_improver(name: str):
    """Parameter-free averaging improver by method name (``history_average`` or ``ema_<decay>``)."""
    if name == "history_average":
        return HistoryAverage()
    if name.startswith("ema_"):
        return EMAWeights(float(name[len("ema_"):]))
    raise KeyError(f"unknown averaging method {name!r}")
