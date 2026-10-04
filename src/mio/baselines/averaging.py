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
