"""Conventional training on flat parameter vectors (AdamW).

Data order is *stateless*: the minibatch used at data-step ``k`` under order
seed ``s`` is a pure function of ``(s, k)``. Together with a saved AdamW state
this makes any checkpoint exactly resumable without pickling RNG state, and
lets baselines replay or diverge from a recorded trajectory deliberately.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Callable

import torch
import torch.nn.functional as F

from mio.config import OptimizerConfig
from mio.datasets.tasks import Split
from mio.state import OptState


@dataclass(frozen=True)
class TrainSettings:
    lr: float
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.0
    batch_size: int = 128
    order_seed: int = 0
    lr_schedule: str = "constant"  # constant | linear_decay
    decay_steps: int | None = None
    class_weights: tuple[float, ...] | None = None

    @classmethod
    def from_config(cls, cfg: OptimizerConfig, order_seed: int) -> "TrainSettings":
        return cls(
            lr=cfg.lr,
            beta1=cfg.beta1,
            beta2=cfg.beta2,
            eps=cfg.eps,
            weight_decay=cfg.weight_decay,
            batch_size=cfg.batch_size,
            order_seed=order_seed,
        )

    def lr_at(self, k: int) -> float:
        if self.lr_schedule == "constant":
            return self.lr
        if self.lr_schedule == "linear_decay":
            if not self.decay_steps:
                raise ValueError("linear_decay requires decay_steps")
            return self.lr * max(0.0, 1.0 - k / self.decay_steps)
        raise ValueError(f"unknown lr schedule {self.lr_schedule!r}")

    def with_(self, **kw) -> "TrainSettings":
        return replace(self, **kw)

    def describe(self) -> dict:
        return {
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "lr_schedule": self.lr_schedule,
            "decay_steps": self.decay_steps,
            "class_weights": list(self.class_weights) if self.class_weights else None,
            "order_seed": self.order_seed,
        }


class BatchSampler:
    """Stateless epoch-permutation sampler (drops the last partial batch)."""

    def __init__(self, n: int, batch_size: int, order_seed: int):
        if n < batch_size:
            raise ValueError(f"dataset of {n} examples is smaller than batch size {batch_size}")
        self.n = n
        self.batch_size = batch_size
        self.order_seed = order_seed
        self.steps_per_epoch = n // batch_size
        self._epoch = -1
        self._perm: torch.Tensor | None = None

    def indices(self, k: int) -> torch.Tensor:
        epoch, pos = divmod(k, self.steps_per_epoch)
        if epoch != self._epoch:
            g = torch.Generator().manual_seed((self.order_seed * 1_000_003 + epoch) % (2**63))
            self._perm = torch.randperm(self.n, generator=g)
            self._epoch = epoch
        assert self._perm is not None
        return self._perm[pos * self.batch_size : (pos + 1) * self.batch_size]


def adamw_update(
    theta: torch.Tensor, grad: torch.Tensor, opt: OptState, lr: float, beta1: float, beta2: float, eps: float, wd: float
) -> None:
    """In-place AdamW step, matching ``torch.optim.AdamW`` (amsgrad=False)."""
    opt.step += 1
    theta.mul_(1.0 - lr * wd)
    opt.m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
    opt.v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
    bc1 = 1.0 - beta1**opt.step
    bc2 = 1.0 - beta2**opt.step
    denom = (opt.v.sqrt() / (bc2**0.5)).add_(eps)
    theta.addcdiv_(opt.m, denom, value=-lr / bc1)


def adam_direction(opt: OptState, beta1: float, beta2: float, eps: float) -> torch.Tensor:
    """Bias-corrected Adam update direction m_hat / (sqrt(v_hat) + eps) (no lr, no sign)."""
    if opt.step == 0:
        return torch.zeros_like(opt.m)
    bc1 = 1.0 - beta1**opt.step
    bc2 = 1.0 - beta2**opt.step
    return (opt.m / bc1) / ((opt.v / bc2).sqrt() + eps)


def batch_loss(model, theta: torch.Tensor, x: torch.Tensor, y: torch.Tensor, class_weights: torch.Tensor | None):
    return F.cross_entropy(model.forward(theta, x), y, weight=class_weights)


@dataclass
class SegmentResult:
    theta: torch.Tensor
    opt: OptState
    steps: int
    wall_time_s: float
    losses: list[float] = field(default_factory=list)


def train_steps(
    model,
    theta: torch.Tensor,
    opt: OptState,
    data: Split,
    settings: TrainSettings,
    n_steps: int,
    data_step_start: int = 0,
    callback: Callable[[int, torch.Tensor, OptState], None] | None = None,
    record_loss_every: int = 0,
) -> SegmentResult:
    """Run ``n_steps`` AdamW steps. Inputs are not modified.

    ``callback(k, theta, opt)`` is called after each step with ``k`` = steps
    completed in this segment (1-based).
    """
    theta = theta.detach().clone()
    opt = opt.clone()
    sampler = BatchSampler(len(data), settings.batch_size, settings.order_seed)
    cw = torch.tensor(settings.class_weights, dtype=torch.float32) if settings.class_weights else None
    losses: list[float] = []
    t0 = time.perf_counter()
    for k in range(n_steps):
        idx = sampler.indices(data_step_start + k)
        p = theta.requires_grad_(True)
        loss = batch_loss(model, p, data.x[idx], data.y[idx], cw)
        (grad,) = torch.autograd.grad(loss, p)
        theta = p.detach()
        with torch.no_grad():
            adamw_update(
                theta, grad, opt, settings.lr_at(k), settings.beta1, settings.beta2, settings.eps, settings.weight_decay
            )
        if record_loss_every and (k + 1) % record_loss_every == 0:
            losses.append(float(loss))
        if callback is not None:
            callback(k + 1, theta, opt)
    return SegmentResult(theta, opt, n_steps, time.perf_counter() - t0, losses)


class EMATracker:
    """Bias-corrected exponential moving averages of the iterates, for several decays.

    ``raw_d <- d * raw_d + (1 - d) * theta_k`` after every step ``k`` (zero-initialized), and
    ``ema_d(t) = raw_d / (1 - d^t)``: an exponentially weighted mean of theta_1..theta_t. A tracker
    can resume from stored bias-corrected EMAs at step ``t`` (``from_corrected``).
    """

    def __init__(self, decays, n: int, step: int = 0):
        self.raw = {float(d): torch.zeros(n) for d in decays}
        self.step = step

    @classmethod
    def from_corrected(cls, ema: dict[float, torch.Tensor], step: int) -> "EMATracker":
        t = cls([], 0, step)
        t.raw = {float(d): e * (1.0 - d**step) for d, e in ema.items()}
        return t

    def update(self, theta: torch.Tensor) -> None:
        self.step += 1
        for d, e in self.raw.items():
            e.mul_(d).add_(theta, alpha=1.0 - d)

    def corrected(self) -> dict[float, torch.Tensor]:
        if self.step == 0:
            return {}
        return {d: e / (1.0 - d**self.step) for d, e in self.raw.items()}
