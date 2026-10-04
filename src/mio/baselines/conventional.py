"""Reference and conventional-optimization baselines."""

from __future__ import annotations

import time

import torch

from mio.datasets.tasks import Split
from mio.state import Condition, Cost, ModelState, OptState, Proposal
from mio.trajectories.training import TrainSettings, train_steps
from mio.utils.reproducibility import derive_seed


class NoUpdate:
    name = "no_update"

    def propose(self, state: ModelState, condition: Condition) -> Proposal:
        return Proposal(delta=torch.zeros_like(state.theta))


def settings_from_state(state: ModelState, order_seed: int, beta1: float, beta2: float, eps: float) -> TrainSettings:
    hp = state.hparams
    return TrainSettings(
        lr=hp["lr"],
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        weight_decay=hp["weight_decay"],
        batch_size=hp["batch_size"],
        order_seed=order_seed,
    )


class ContinuedAdamW:
    """Continue AdamW from the parent's exact optimizer state.

    ``steps=None`` runs ``condition.horizon`` steps (the reference optimizer at
    the requested horizon); an integer runs a fixed budget (e.g. matched to
    the learned operator's per-application compute). A fresh data order is
    derived from ``seed`` and the parent's checkpoint id.
    """

    def __init__(self, model, train: Split, beta1: float, beta2: float, eps: float, steps: int | None = None,
                 seed: int = 0, name: str | None = None, order_seed: int | None = None):
        self.model = model
        self.train = train
        self.beta1, self.beta2, self.eps = beta1, beta2, eps
        self.steps = steps
        self.seed = seed
        self.fixed_order_seed = order_seed
        self.name = name or ("adamw_full" if steps is None else f"adamw_{steps}")

    def order_seed(self, state: ModelState) -> int:
        if self.fixed_order_seed is not None:
            return self.fixed_order_seed
        return derive_seed(self.seed, "adamw-baseline", state.checkpoint_id or "anon")

    def propose(self, state: ModelState, condition: Condition) -> Proposal:
        if state.opt is None:
            raise ValueError("ContinuedAdamW needs the parent's optimizer state")
        n = condition.horizon if self.steps is None else self.steps
        settings = settings_from_state(state, self.order_seed(state), self.beta1, self.beta2, self.eps)
        t0 = time.perf_counter()
        res = train_steps(self.model, state.theta, state.opt, self.train, settings, n)
        return Proposal(
            delta=res.theta - state.theta,
            cost=Cost(task_fb_passes=n, fb_equivalent=float(n), wall_time_s=time.perf_counter() - t0),
            info={"steps": n, "order_seed": settings.order_seed},
        )


class ClassWeightedAdamW(ContinuedAdamW):
    """Steering baseline: continue AdamW with the *requested* class up-weighted.

    Reads the requested objective ``class_k`` from the condition and trains ``steps`` steps with
    loss weight ``weight`` on class k (1 elsewhere), i.e. the population's ``class_weight``
    intervention. The proposal ignores the requested magnitude and the horizon. The data order is
    the same as ``ContinuedAdamW``'s, so budgets of one weight are prefixes of one run.
    """

    uses_horizon = False

    def __init__(self, model, train: Split, beta1: float, beta2: float, eps: float, steps: int, weight: float,
                 num_classes: int, seed: int = 0, name: str | None = None):
        super().__init__(model, train, beta1, beta2, eps, steps=steps, seed=seed,
                         name=name or f"class_ft_w{weight:g}_n{steps}")
        self.weight = weight
        self.num_classes = num_classes

    def propose(self, state: ModelState, condition: Condition) -> Proposal:
        if state.opt is None:
            raise ValueError("ClassWeightedAdamW needs the parent's optimizer state")
        classes = [int(o.split("_")[1]) for o in condition.gains if o.startswith("class_")]
        if len(classes) != 1:
            raise ValueError(f"ClassWeightedAdamW needs exactly one class request, got {condition.gains}")
        w = [1.0] * self.num_classes
        w[classes[0]] = self.weight
        settings = settings_from_state(state, self.order_seed(state), self.beta1, self.beta2, self.eps)
        settings = settings.with_(class_weights=tuple(w))
        t0 = time.perf_counter()
        res = train_steps(self.model, state.theta, state.opt, self.train, settings, self.steps)
        return Proposal(
            delta=res.theta - state.theta,
            cost=Cost(task_fb_passes=self.steps, fb_equivalent=float(self.steps),
                      wall_time_s=time.perf_counter() - t0),
            info={"steps": self.steps, "weight": self.weight, "target_class": classes[0],
                  "order_seed": settings.order_seed},
        )


def reference_curve(model, state: ModelState, train: Split, evaluate_fn, n_steps: int, every: int,
                    beta1: float, beta2: float, eps: float, order_seed: int) -> list[tuple[int, float]]:
    """Report-split loss along an AdamW continuation from the exact parent state."""
    settings = settings_from_state(state, order_seed, beta1, beta2, eps)
    curve = [(0, evaluate_fn(state.theta))]

    def cb(k: int, theta: torch.Tensor, opt: OptState) -> None:
        if k % every == 0 or k == n_steps:
            curve.append((k, evaluate_fn(theta)))

    train_steps(model, state.theta, state.opt, train, settings, n_steps, callback=cb)
    return curve


def reference_envelope(curve: list[tuple[int, float]], window: int = 1) -> list[float]:
    """Best-so-far loss of the reference optimizer, after light smoothing.

    Point 0 (the parent) is kept exact. Later points are replaced by a centred
    moving average over ``window`` evaluations (restricted to points >= 1) to
    remove evaluation noise, then by their running minimum. The envelope is
    therefore "AdamW with best-checkpoint selection": monotone by construction,
    and correct when the raw curve turns upward late in training (overfitting),
    where a monotone *fit* would pool the rising tail above the true minimum.
    """
    ys = [v for _, v in curve]
    half = max(window, 1) // 2
    smoothed = [ys[0]]
    for i in range(1, len(ys)):
        lo, hi = max(1, i - half), min(len(ys), i + half + 1)
        smoothed.append(sum(ys[lo:hi]) / (hi - lo))
    env, best = [], float("inf")
    for v in smoothed:
        best = min(best, v)
        env.append(best)
    return env


def equivalent_steps(curve: list[tuple[int, float]], child_loss: float, window: int = 1) -> float | None:
    """Reference AdamW steps needed to match ``child_loss`` (see ``reference_envelope``).

    Returns 0 if the child is no better than the parent, ``None`` if the
    reference never reaches it within the curve (a candidate *enhancement*),
    and interpolates linearly between evaluation points otherwise.
    """
    if child_loss >= curve[0][1]:  # no gain over the parent's exact loss
        return 0.0
    ks = [k for k, _ in curve]
    env = reference_envelope(curve, window)
    for i in range(1, len(env)):
        if env[i] <= child_loss:
            frac = (env[i - 1] - child_loss) / max(env[i - 1] - env[i], 1e-12)
            return float(ks[i - 1] + frac * (ks[i] - ks[i - 1]))
    return None
