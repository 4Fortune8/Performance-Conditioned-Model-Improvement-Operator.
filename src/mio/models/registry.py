"""Model registry: build a functional target model from config + task shape."""

from __future__ import annotations

from typing import Callable

from mio.config import ModelConfig
from mio.models.base import MLP

_REGISTRY: dict[str, Callable[[ModelConfig, int, int], object]] = {
    "mlp": lambda cfg, input_dim, num_classes: MLP(input_dim, cfg.hidden, num_classes),
}


def build_model(cfg: ModelConfig, input_dim: int, num_classes: int) -> MLP:
    if cfg.name not in _REGISTRY:
        raise KeyError(f"unknown model {cfg.name!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[cfg.name](cfg, input_dim, num_classes)


def register_model(name: str, builder: Callable[[ModelConfig, int, int], object]) -> None:
    _REGISTRY[name] = builder
