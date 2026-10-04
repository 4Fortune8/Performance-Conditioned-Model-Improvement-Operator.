"""Per-parameter feature construction for coordinatewise operators.

Every feature is either a per-coordinate quantity or a tensor-level aggregate
broadcast to the tensor's coordinates, so any operator built on these features
is equivariant to permutations of hidden units (by construction, with no
alignment step). Feature groups double as input ablations:

    weights    theta_i / rms_T(theta), log rms_T(theta)
    history    (theta - theta_{t-L})_i / rms_T(.), log(rms_T(.) / L) for each lag L
    adam       bias-corrected Adam direction m_hat / (sqrt(v_hat) + eps)
    metrics    accept-split loss and accuracy of the parent (global)
    condition  requested gains + presence mask per objective (global)

Structural features (tensor position one-hot), log horizon, log step and log
lr are always present.

Targets live in Adam's natural unit: ``y = delta / (lr * horizon)``. Under
Adam each step moves a coordinate by at most ~lr, so ``|y| <= ~1`` for any
consistent motion; the unit needs no trajectory information, so the
weights-only ablation stays trajectory-free.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from mio.models.base import ParamSpec
from mio.state import Condition, ModelState
from mio.trajectories.training import adam_direction

ADAM_CLIP = 10.0
TINY = 1e-12


def _role_onehot(spec: ParamSpec) -> torch.Tensor:
    """(N, 6): (first | middle | last layer) x (weight | bias)."""
    n_layers = spec.num_layers
    cols = []
    for t in spec.tensors:
        pos = 0 if t.layer_index == 0 else (2 if t.layer_index == n_layers - 1 else 1)
        v = torch.zeros(6)
        v[pos * 2 + (0 if t.role == "weight" else 1)] = 1.0
        cols.append(v.expand(t.numel, 6))
    return torch.cat(cols)


@dataclass
class FeatureSpec:
    groups: tuple[str, ...]
    lags: tuple[int, ...]
    objectives: tuple[str, ...]
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8

    @property
    def n_param(self) -> int:
        n = 6  # role one-hot
        if "weights" in self.groups:
            n += 2
        if "history" in self.groups:
            n += 2 * len(self.lags)
        if "adam" in self.groups:
            n += 1
        return n

    @property
    def n_global(self) -> int:
        n = 3  # log horizon, log step, log lr
        if "metrics" in self.groups:
            n += 2
        if "condition" in self.groups:
            n += 2 * len(self.objectives)
        return n

    def to_dict(self) -> dict:
        return {
            "groups": list(self.groups),
            "lags": list(self.lags),
            "objectives": list(self.objectives),
            "beta1": self.beta1,
            "beta2": self.beta2,
            "eps": self.eps,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureSpec":
        return cls(tuple(d["groups"]), tuple(d["lags"]), tuple(d["objectives"]), d["beta1"], d["beta2"], d["eps"])


class FeatureBuilder:
    def __init__(self, spec: ParamSpec, fspec: FeatureSpec):
        self.spec = spec
        self.fspec = fspec
        self._roles = _role_onehot(spec)

    def param_features(self, state: ModelState) -> torch.Tensor:
        """(N, n_param) per-coordinate features."""
        spec, fs = self.spec, self.fspec
        cols: list[torch.Tensor] = [self._roles]
        if "weights" in fs.groups:
            rms = spec.per_tensor_rms(state.theta) + TINY
            cols.append((state.theta / spec.expand(rms)).unsqueeze(1))
            cols.append(spec.expand(rms.log()).unsqueeze(1))
        if "history" in fs.groups:
            for lag in fs.lags:
                if lag not in state.history:
                    raise KeyError(f"state is missing history at lag {lag}")
                d = state.theta - state.history[lag]
                rms = spec.per_tensor_rms(d) + TINY
                cols.append((d / spec.expand(rms)).unsqueeze(1))
                cols.append(spec.expand((rms / lag).log()).unsqueeze(1))
        if "adam" in fs.groups:
            if state.opt is None:
                raise ValueError("adam features requested but state has no optimizer state")
            u = adam_direction(state.opt, fs.beta1, fs.beta2, fs.eps).clamp(-ADAM_CLIP, ADAM_CLIP)
            cols.append(u.unsqueeze(1))
        return torch.cat(cols, dim=1)

    def global_features(self, state: ModelState, condition: Condition, gain_scale: dict[str, float],
                        mask: dict[str, bool] | None = None) -> torch.Tensor:
        """(n_global,) features shared by all coordinates.

        ``mask`` optionally hides objectives that are present in the condition
        (used for condition dropout during training).
        """
        fs = self.fspec
        lr = float(state.hparams.get("lr", 1e-3))
        vals = [math.log(condition.horizon), math.log(state.step + 1), math.log(lr)]
        if "metrics" in fs.groups:
            vals += [float(state.metrics["loss"]), float(state.metrics["acc"])]
        if "condition" in fs.groups:
            for obj in fs.objectives:
                present = obj in condition.gains and (mask is None or mask.get(obj, True))
                vals += [1.0 if present else 0.0, condition.gains[obj] / gain_scale[obj] if present else 0.0]
        return torch.tensor(vals, dtype=torch.float32)

    @staticmethod
    def target_scale(state: ModelState, horizon: int) -> float:
        return float(state.hparams.get("lr", 1e-3)) * horizon
