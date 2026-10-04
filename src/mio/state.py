"""Shared value types passed between generation, operators and evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class OptState:
    """AdamW state over the flat parameter vector."""

    m: torch.Tensor
    v: torch.Tensor
    step: int

    @classmethod
    def zeros(cls, n: int) -> "OptState":
        return cls(torch.zeros(n), torch.zeros(n), 0)

    def clone(self) -> "OptState":
        return OptState(self.m.clone(), self.v.clone(), self.step)


@dataclass
class ModelState:
    """Everything an improver may observe about a parent model.

    ``history`` maps a step lag L to the parameters at ``step - L`` (the
    trajectory). ``ema`` maps a decay to the bias-corrected exponential moving
    average of the iterates up to ``step`` (empty if not tracked). ``metrics``
    are measured on the *accept* split only. ``hparams`` holds the optimizer
    settings in effect at ``step``.
    """

    theta: torch.Tensor
    step: int
    history: dict[int, torch.Tensor] = field(default_factory=dict)
    opt: OptState | None = None
    ema: dict[float, torch.Tensor] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    hparams: dict[str, Any] = field(default_factory=dict)
    checkpoint_id: str | None = None
    root_id: str | None = None
    group_id: str | None = None


@dataclass
class Condition:
    """Requested improvement.

    ``gains`` maps objective names to requested *gains* (positive = better),
    e.g. ``{"loss": 0.05}`` or ``{"class_3": 0.2}``. Objectives not present are
    unconstrained ("don't care"). ``horizon`` is the step-equivalent size of
    the requested move.
    """

    horizon: int
    gains: dict[str, float] = field(default_factory=dict)


@dataclass
class Cost:
    """Hardware-independent cost of producing a proposal.

    ``task_fb_passes`` counts forward+backward passes over a training batch
    (one optimizer step == 1). ``operator_flops`` counts the improver's own
    compute; ``fb_equivalent`` converts everything into batch fb passes.
    """

    task_fb_passes: float = 0.0
    task_forward_examples: int = 0
    operator_flops: float = 0.0
    wall_time_s: float = 0.0
    fb_equivalent: float = 0.0


@dataclass
class Proposal:
    delta: torch.Tensor
    cost: Cost = field(default_factory=Cost)
    info: dict[str, Any] = field(default_factory=dict)
