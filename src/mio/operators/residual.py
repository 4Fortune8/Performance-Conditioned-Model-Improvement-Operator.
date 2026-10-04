"""Coordinatewise residual improvement operator.

A small MLP shared by every parameter coordinate (Introspection / VeLO style)
predicts the normalized update ``y_i`` for coordinate ``i`` from its own
features and a global context vector (metrics, horizon, requested gains):

    delta_i = y_i * lr * horizon

Because the same function is applied to every coordinate and features are
per-coordinate or tensor aggregates, the operator commutes with permutations
of hidden units. It can also be applied to networks of other widths.

Conditioning uses hindsight relabeling: during training, the condition is the
outcome that the observed transition actually achieved (with random objective
dropout); at inference, the caller requests the outcome it wants.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from mio.models.base import ParamSpec
from mio.operators.features import FeatureBuilder, FeatureSpec
from mio.state import Condition, Cost, ModelState, Proposal


class CoordinatewiseNet(nn.Module):
    def __init__(self, n_param: int, n_global: int, hidden: list[int]):
        super().__init__()
        if not hidden:
            raise ValueError("hidden must have at least one layer")
        self.n_param, self.n_global, self.hidden = n_param, n_global, list(hidden)
        self.param_in = nn.Linear(n_param, hidden[0])
        self.global_in = nn.Linear(n_global, hidden[0], bias=False)
        layers: list[nn.Module] = []
        for a, b in zip(hidden[:-1], hidden[1:]):
            layers += [nn.GELU(), nn.Linear(a, b)]
        layers += [nn.GELU(), nn.Linear(hidden[-1], 1)]
        self.body = nn.Sequential(*layers)
        nn.init.zeros_(self.body[-1].weight)  # start as the no-update operator
        nn.init.zeros_(self.body[-1].bias)
        # Input standardization (set from training data before fitting).
        self.register_buffer("p_mean", torch.zeros(n_param))
        self.register_buffer("p_std", torch.ones(n_param))
        self.register_buffer("g_mean", torch.zeros(n_global))
        self.register_buffer("g_std", torch.ones(n_global))

    def set_normalization(self, xp: torch.Tensor, xg: torch.Tensor) -> None:
        def stats(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            mean, std = x.mean(0), x.std(0)
            const = std < 1e-6
            return torch.where(const, torch.zeros_like(mean), mean), torch.where(const, torch.ones_like(std), std)

        self.p_mean, self.p_std = stats(xp)
        self.g_mean, self.g_std = stats(xg)

    def forward(self, xp: torch.Tensor, xg: torch.Tensor, group: torch.Tensor | None = None) -> torch.Tensor:
        """``xp``: (M, n_param); ``xg``: (G, n_global); ``group``: (M,) row -> context index.

        With ``group=None``, ``xg`` must be a single context (1, n_global) or (n_global,).
        """
        hp = self.param_in((xp - self.p_mean) / self.p_std)
        hg = self.global_in((xg.reshape(-1, self.n_global) - self.g_mean) / self.g_std)
        h = hp + (hg[group] if group is not None else hg)
        return self.body(h).squeeze(-1)

    def flops_per_coordinate(self) -> float:
        f = 2 * self.n_param * self.hidden[0]
        for a, b in zip(self.hidden[:-1], self.hidden[1:]):
            f += 2 * a * b
        f += 2 * self.hidden[-1]
        return float(f)


class LearnedOperator:
    """``Improver`` wrapper around a trained ``CoordinatewiseNet``."""

    def __init__(self, net: CoordinatewiseNet, spec: ParamSpec, fspec: FeatureSpec, gain_scale: dict[str, float],
                 name: str = "operator", batch_size_for_cost: int = 128):
        self.net = net.eval()
        self.spec = spec
        self.fspec = fspec
        self.builder = FeatureBuilder(spec, fspec)
        self.gain_scale = gain_scale
        self.name = name
        self.batch_size_for_cost = batch_size_for_cost

    @torch.no_grad()
    def predict_normalized(self, state: ModelState, condition: Condition) -> torch.Tensor:
        xp = self.builder.param_features(state)
        xg = self.builder.global_features(state, condition, self.gain_scale)
        return self.net(xp, xg)

    @torch.no_grad()
    def propose(self, state: ModelState, condition: Condition) -> Proposal:
        t0 = time.perf_counter()
        y = self.predict_normalized(state, condition)
        delta = y * FeatureBuilder.target_scale(state, condition.horizon)
        flops = self.flops()
        return Proposal(
            delta=delta,
            cost=Cost(
                operator_flops=flops,
                wall_time_s=time.perf_counter() - t0,
                fb_equivalent=flops / (6.0 * self.spec.numel * self.batch_size_for_cost),
            ),
            info={"condition": dict(condition.gains)},
        )

    def flops(self) -> float:
        return self.net.flops_per_coordinate() * self.spec.numel

    # ---- persistence ----------------------------------------------------
    def save(self, path: str | Path, config: dict, training_summary: dict) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "net_state": self.net.state_dict(),
                "hidden": self.net.hidden,
                "feature_spec": self.fspec.to_dict(),
                "gain_scale": self.gain_scale,
                "structure_hash": self.spec.structure_hash,
                "config": config,
                "training_summary": training_summary,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, spec: ParamSpec, name: str = "operator", check_structure: bool = True
             ) -> "LearnedOperator":
        d = torch.load(path, map_location="cpu", weights_only=False)
        if check_structure and d["structure_hash"] != spec.structure_hash:
            # Coordinatewise operators *can* run on other widths; require opt-in.
            raise ValueError(
                f"operator trained on structure {d['structure_hash']} but target is {spec.structure_hash}; "
                "pass check_structure=False to apply it across architectures deliberately"
            )
        fspec = FeatureSpec.from_dict(d["feature_spec"])
        net = CoordinatewiseNet(fspec.n_param, fspec.n_global, d["hidden"])
        net.load_state_dict(d["net_state"])
        op = cls(net, spec, fspec, d["gain_scale"], name=name)
        op.training_summary = d.get("training_summary", {})
        return op


class ConditionPolicy:
    """Turns "improve objective X" into a concrete requested gain.

    The request is the ``quantile`` of gains achieved by the ``k`` training
    transitions closest to the parent in (log source step, horizon). This asks
    for an outcome that is ambitious but within the training distribution.
    """

    def __init__(self, transitions: list[dict], metric_split: str = "accept"):
        self.steps = np.array([r["source_step"] for r in transitions], dtype=float)
        self.horizons = np.array([r["horizon"] for r in transitions], dtype=float)
        self.gains = [r["gains"][metric_split] for r in transitions]
        if not len(self.steps):
            raise ValueError("ConditionPolicy needs at least one training transition")

    def request(self, state: ModelState, horizon: int, objective: str = "loss", quantile: float = 0.75,
                k: int = 50) -> Condition:
        d_step = np.abs(np.log1p(self.steps) - np.log1p(state.step))
        d_h = np.abs(np.log(self.horizons) - np.log(horizon))
        order = np.lexsort((d_step, d_h))  # same horizon first, then nearest step
        nearest = order[: min(k, len(order))]
        value = float(np.quantile([self.gains[i][objective] for i in nearest], quantile))
        return Condition(horizon=horizon, gains={objective: value})
