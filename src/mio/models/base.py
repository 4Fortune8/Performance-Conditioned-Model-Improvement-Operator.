"""Functional target models over flat parameter vectors.

Every model in a population is represented by a single flat float32 vector
``theta`` plus a ``ParamSpec`` describing how that vector is cut into tensors.
Working with flat vectors makes deltas, interpolation, optimizer state and
operator inputs trivially aligned, and makes the forward pass a pure function
of ``theta`` (needed later for behavioural / differentiable operator training).
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F


class IncompatibleModelError(ValueError):
    """Raised when parameters do not match the expected model structure."""


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple[int, ...]
    offset: int
    layer_index: int  # 0-based linear layer index
    role: str  # "weight" | "bias"
    fan_in: int
    fan_out: int

    @property
    def numel(self) -> int:
        return math.prod(self.shape)

    @property
    def slice(self) -> slice:
        return slice(self.offset, self.offset + self.numel)


@dataclass(frozen=True)
class ParamSpec:
    model_name: str
    tensors: tuple[TensorSpec, ...]

    @property
    def numel(self) -> int:
        return sum(t.numel for t in self.tensors)

    @property
    def num_layers(self) -> int:
        return 1 + max(t.layer_index for t in self.tensors)

    @property
    def structure_hash(self) -> str:
        payload = json.dumps([self.model_name, [(t.name, list(t.shape)) for t in self.tensors]])
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        return {"model_name": self.model_name, "tensors": [asdict(t) for t in self.tensors]}

    @classmethod
    def from_dict(cls, d: dict) -> "ParamSpec":
        return cls(d["model_name"], tuple(TensorSpec(**{**t, "shape": tuple(t["shape"])}) for t in d["tensors"]))

    def unflatten(self, theta: torch.Tensor) -> dict[str, torch.Tensor]:
        self.check_vector(theta, check_finite=False)
        return {t.name: theta[t.slice].view(t.shape) for t in self.tensors}

    def flatten(self, tensors: dict[str, torch.Tensor]) -> torch.Tensor:
        missing = [t.name for t in self.tensors if t.name not in tensors]
        if missing:
            raise IncompatibleModelError(f"missing tensors {missing}")
        parts = []
        for t in self.tensors:
            x = tensors[t.name]
            if tuple(x.shape) != t.shape:
                raise IncompatibleModelError(f"{t.name}: expected shape {t.shape}, got {tuple(x.shape)}")
            parts.append(x.reshape(-1))
        return torch.cat(parts)

    def check_vector(self, theta: torch.Tensor, check_finite: bool = True) -> None:
        if theta.dim() != 1 or theta.numel() != self.numel:
            raise IncompatibleModelError(
                f"{self.model_name}: expected flat vector of {self.numel} params, got shape {tuple(theta.shape)}"
            )
        if check_finite and not torch.isfinite(theta).all():
            raise IncompatibleModelError("parameter vector contains NaN or Inf")

    def check_compatible(self, other: "ParamSpec") -> None:
        if self.structure_hash != other.structure_hash:
            raise IncompatibleModelError(
                f"incompatible structures: {self.model_name}[{self.structure_hash}] vs "
                f"{other.model_name}[{other.structure_hash}]"
            )

    def tensor_index(self) -> torch.Tensor:
        """Long tensor mapping each flat coordinate to its tensor index."""
        return torch.cat([torch.full((t.numel,), i, dtype=torch.long) for i, t in enumerate(self.tensors)])

    def per_tensor_rms(self, v: torch.Tensor, eps: float = 0.0) -> torch.Tensor:
        """RMS of ``v`` within each tensor, returned as a per-tensor vector."""
        return torch.stack([v[t.slice].pow(2).mean().add(eps).sqrt() for t in self.tensors])

    def per_tensor_norm(self, v: torch.Tensor) -> torch.Tensor:
        return torch.stack([v[t.slice].norm() for t in self.tensors])

    def expand(self, per_tensor: torch.Tensor) -> torch.Tensor:
        """Broadcast a per-tensor vector to a flat per-coordinate vector."""
        return torch.cat([per_tensor[i].expand(t.numel) for i, t in enumerate(self.tensors)])


class MLP:
    """Fully connected ReLU classifier evaluated directly from a flat vector."""

    def __init__(self, input_dim: int, hidden: list[int], num_classes: int):
        self.input_dim = input_dim
        self.hidden = list(hidden)
        self.num_classes = num_classes
        dims = [input_dim, *self.hidden, num_classes]
        specs: list[TensorSpec] = []
        offset = 0
        for i, (din, dout) in enumerate(zip(dims[:-1], dims[1:])):
            for role, shape in (("weight", (dout, din)), ("bias", (dout,))):
                spec = TensorSpec(f"layers.{i}.{role}", shape, offset, i, role, din, dout)
                specs.append(spec)
                offset += spec.numel
        arch = "x".join(str(d) for d in dims)
        self.spec = ParamSpec(f"mlp_{arch}", tuple(specs))

    @property
    def num_params(self) -> int:
        return self.spec.numel

    def init(self, generator: torch.Generator) -> torch.Tensor:
        """PyTorch nn.Linear default init: U(-1/sqrt(fan_in), 1/sqrt(fan_in))."""
        theta = torch.empty(self.spec.numel)
        for t in self.spec.tensors:
            bound = 1.0 / math.sqrt(t.fan_in)
            theta[t.slice] = torch.rand(t.numel, generator=generator) * 2 * bound - bound
        return theta

    def forward(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        p = self.spec.unflatten(theta)
        n_layers = self.spec.num_layers
        h = x
        for i in range(n_layers):
            h = F.linear(h, p[f"layers.{i}.weight"], p[f"layers.{i}.bias"])
            if i < n_layers - 1:
                h = F.relu(h)
        return h

    def permute_hidden(self, theta: torch.Tensor, layer: int, perm: torch.Tensor) -> torch.Tensor:
        """Permute the output units of hidden ``layer`` (function-preserving).

        Used to test permutation equivariance of operators and for alignment.
        """
        if not 0 <= layer < self.spec.num_layers - 1:
            raise ValueError("can only permute hidden layers")
        p = {k: v.clone() for k, v in self.spec.unflatten(theta).items()}
        p[f"layers.{layer}.weight"] = p[f"layers.{layer}.weight"][perm]
        p[f"layers.{layer}.bias"] = p[f"layers.{layer}.bias"][perm]
        p[f"layers.{layer + 1}.weight"] = p[f"layers.{layer + 1}.weight"][:, perm]
        return self.spec.flatten(p)
