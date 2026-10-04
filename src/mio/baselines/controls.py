"""Negative controls and the oracle.

- ``RandomNormMatched``: random direction with per-tensor norms matched to a
  reference method's delta. Any "improvement" from it is noise or a
  norm-only effect.
- ``ForeignDelta``: the real recorded delta of a *different* (training-root)
  model at a similar step and the same horizon, applied without alignment.
  Tests whether coordinate-level update structure transfers across
  initializations at all.
- ``Oracle``: the true recorded delta to a given target checkpoint (round-trip
  correctness test and upper bound).
"""

from __future__ import annotations

import numpy as np
import torch

from mio.models.base import ParamSpec
from mio.state import Condition, ModelState, Proposal
from mio.trajectories.checkpoints import CheckpointStore
from mio.utils.reproducibility import derive_seed


class RandomNormMatched:
    def __init__(self, spec: ParamSpec, reference, seed: int = 0, name: str = "random_norm_matched"):
        self.spec = spec
        self.reference = reference
        self.seed = seed
        self.name = name

    def propose(self, state: ModelState, condition: Condition) -> Proposal:
        ref = self.reference.propose(state, condition).delta
        g = torch.Generator().manual_seed(derive_seed(self.seed, "random-control", state.checkpoint_id or "anon",
                                                      condition.horizon))
        noise = torch.randn(ref.numel(), generator=g)
        scale = self.spec.per_tensor_norm(ref) / (self.spec.per_tensor_norm(noise) + 1e-30)
        return Proposal(delta=noise * self.spec.expand(scale), info={"reference": self.reference.name})


class ForeignDelta:
    def __init__(self, store: CheckpointStore, train_transitions: list[dict], name: str = "foreign_delta"):
        self.store = store
        self.transitions = [t for t in train_transitions if t["target_line"] == "trunk"] or train_transitions
        self.name = name

    def propose(self, state: ModelState, condition: Condition) -> Proposal:
        cands = [t for t in self.transitions if t["root_id"] != state.root_id]
        if not cands:
            raise ValueError("no foreign transitions available")
        steps = np.array([t["source_step"] for t in cands], dtype=float)
        hs = np.array([t["horizon"] for t in cands], dtype=float)
        order = np.lexsort((np.abs(np.log1p(steps) - np.log1p(state.step)), np.abs(np.log(hs) - np.log(condition.horizon))))
        t = cands[int(order[0])]
        delta = self.store.theta(t["target_id"]) - self.store.theta(t["source_id"])
        # Rescale linearly if the closest recorded horizon differs from the requested one.
        delta = delta * (condition.horizon / t["horizon"])
        return Proposal(delta=delta, info={"foreign_transition": t["id"]})


class Oracle:
    """Applies the recorded delta to a specific target checkpoint."""

    name = "oracle"

    def __init__(self, store: CheckpointStore):
        self.store = store
        self.target_id: str | None = None

    def propose(self, state: ModelState, condition: Condition) -> Proposal:
        if self.target_id is None:
            raise ValueError("set Oracle.target_id before proposing")
        return Proposal(delta=self.store.theta(self.target_id) - state.theta, info={"target": self.target_id})
