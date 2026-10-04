"""The ``Improver`` interface shared by learned operators and baselines.

An improver maps (parent state, requested condition) to a proposed parameter
delta plus the compute it spent. Evaluation never special-cases a method: the
same child engine sanitizes, applies, measures and gates every proposal.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from mio.state import Condition, ModelState, Proposal


@runtime_checkable
class Improver(Protocol):
    name: str

    def propose(self, state: ModelState, condition: Condition) -> Proposal: ...


def adam_step_flops(num_params: int, batch_size: int) -> float:
    """Approximate FLOPs of one forward+backward pass of an MLP on a batch."""
    return 6.0 * num_params * batch_size
