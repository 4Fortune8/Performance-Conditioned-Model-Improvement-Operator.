"""Randomized branch interventions.

At each branch point the generator draws interventions from a fixed menu with
fixed probabilities (a randomized design), so intervention effects are not
confounded with the state of the model at the branch point. The draw, the
branch's data-order seed and any targeted class are all derived from the
master seed, the root and the branch point, so they are reproducible.
"""

from __future__ import annotations

import numpy as np

from mio.config import BranchingConfig, InterventionConfig
from mio.trajectories.training import TrainSettings
from mio.utils.reproducibility import derive_seed


def anchor_schedule(total_steps: int, first: int, growth: float, linear_every: int) -> list[int]:
    """Geometric spacing (dense early training) switching to linear spacing."""
    steps = {0, total_steps}
    s = float(first)
    while s < min(linear_every, total_steps):
        steps.add(int(round(s)))
        s *= growth
    k = linear_every
    while k < total_steps:
        steps.add(k)
        k += linear_every
    return sorted(x for x in steps if 0 <= x <= total_steps)


def branch_points(anchors: list[int], cfg: BranchingConfig, total_steps: int, min_history: int) -> list[int]:
    if not cfg.enabled:
        return []
    eligible = [t for t in anchors if t >= max(cfg.min_step, min_history) and t < total_steps]
    return eligible[:: max(1, cfg.every_n_anchors)]


def draw_interventions(
    cfg: BranchingConfig, master_seed: int, root_id: str, step: int, n: int
) -> list[InterventionConfig]:
    rng = np.random.default_rng(derive_seed(master_seed, "branch-draw", root_id, step))
    weights = np.array([iv.weight for iv in cfg.interventions], dtype=float)
    idx = rng.choice(len(cfg.interventions), size=n, replace=True, p=weights / weights.sum())
    return [cfg.interventions[i] for i in idx]


def apply_intervention(
    base: TrainSettings,
    iv: InterventionConfig,
    horizon: int,
    order_seed: int,
    num_classes: int,
    target_class_seed: int,
) -> tuple[TrainSettings, dict]:
    """Return the branch's training settings and a JSON-able description."""
    s = base.with_(order_seed=order_seed)
    desc: dict = {"name": iv.name, "kind": iv.kind}
    if iv.kind == "continue":
        pass
    elif iv.kind == "lr_scale":
        s = s.with_(lr=base.lr * iv.scale)
        desc["scale"] = iv.scale
    elif iv.kind == "wd_scale":
        s = s.with_(weight_decay=base.weight_decay * iv.scale)
        desc["scale"] = iv.scale
    elif iv.kind == "lr_decay":
        s = s.with_(lr_schedule="linear_decay", decay_steps=horizon)
    elif iv.kind == "class_weight":
        k = int(np.random.default_rng(target_class_seed).integers(num_classes))
        w = [1.0] * num_classes
        w[k] = iv.scale
        s = s.with_(class_weights=tuple(w))
        desc.update(scale=iv.scale, target_class=k)
    else:
        raise ValueError(f"unknown intervention kind {iv.kind!r}")
    return s, desc
