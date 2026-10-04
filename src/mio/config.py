"""Experiment configuration: typed dataclasses loaded from YAML.

A config file describes one experiment end to end (task, model, population,
transitions, operator, evaluation). ``inherit: other.yaml`` deep-merges a base
file first, so research configs can override only what differs from the smoke
test. Unknown keys raise immediately rather than being silently ignored.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SyntheticTaskConfig:
    """Teacher-student task: x ~ N(0, I), y = argmax(teacher_mlp(x)). No download needed."""

    input_dim: int = 20
    num_classes: int = 4
    teacher_hidden: int = 32
    n_train: int = 4000
    n_test: int = 1000
    seed: int = 0


@dataclass
class TaskConfig:
    name: str = "fashion_mnist"  # fashion_mnist | mnist | synthetic
    data_dir: str = "data/raw"
    split_seed: int = 0
    n_accept: int = 5000  # carved from the official train set
    n_dev: int = 5000  # carved from the official train set
    n_train_eval: int = 5000  # fixed subset of the training split used for train metrics
    max_train: int | None = None  # optional cap on the training split (smoke tests)
    synthetic: SyntheticTaskConfig = field(default_factory=SyntheticTaskConfig)


@dataclass
class ModelConfig:
    name: str = "mlp"
    hidden: list[int] = field(default_factory=lambda: [64, 32])


@dataclass
class OptimizerConfig:
    lr: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 1e-4
    batch_size: int = 128


@dataclass
class CheckpointScheduleConfig:
    """Anchor checkpoints: geometric spacing early in training, linear later.

    For every anchor at step t, parameter snapshots at t - lag are also saved
    for each lag in ``history_lags`` so trajectory features are defined at fixed
    step lags regardless of anchor spacing.
    """

    first: int = 100
    growth: float = 2.0
    linear_every: int = 800
    history_lags: list[int] = field(default_factory=lambda: [25, 100])


@dataclass
class InterventionConfig:
    name: str = "continue"
    kind: str = "continue"  # continue | lr_scale | wd_scale | lr_decay | class_weight
    weight: float = 1.0  # relative sampling probability
    scale: float = 1.0  # lr_scale / wd_scale factor, or class_weight factor


@dataclass
class BranchingConfig:
    enabled: bool = True
    every_n_anchors: int = 2  # branch at every k-th eligible trunk anchor
    min_step: int = 100  # do not branch before this trunk step
    n_branches: int = 3
    horizons: list[int] = field(default_factory=lambda: [100, 400])  # checkpoints along each branch
    interventions: list[InterventionConfig] = field(
        default_factory=lambda: [
            InterventionConfig(name="continue", kind="continue", weight=2.0),
            InterventionConfig(name="lr_x0.3", kind="lr_scale", scale=0.3),
            InterventionConfig(name="lr_x3", kind="lr_scale", scale=3.0),
            InterventionConfig(name="wd_x10", kind="wd_scale", scale=10.0),
            InterventionConfig(name="lr_decay", kind="lr_decay"),
            InterventionConfig(name="class_x4", kind="class_weight", scale=4.0),
        ]
    )


@dataclass
class SplitConfig:
    """Root-level partition. The unit is the init group: everything descending
    from one initialization (all order variants, anchors and branches) lands in
    the same partition."""

    train: float = 0.6
    val: float = 0.2
    test: float = 0.2
    seed: int = 0


@dataclass
class PopulationConfig:
    name: str = "smoke"
    n_groups: int = 8  # independent initializations
    order_variants: int = 1  # data-order seeds per initialization (generalization ladder L1)
    master_seed: int = 0
    total_steps: int = 1200
    workers: int = 1
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    checkpoints: CheckpointScheduleConfig = field(default_factory=CheckpointScheduleConfig)
    branching: BranchingConfig = field(default_factory=BranchingConfig)
    split: SplitConfig = field(default_factory=SplitConfig)


@dataclass
class TransitionConfig:
    max_horizon: int = 800
    include_trunk: bool = True
    include_branches: bool = True
    metric_split: str = "accept"  # split whose metric changes label transitions / condition the operator
    success_threshold: float = 0.0  # loss gain above which a transition counts as successful


@dataclass
class OperatorConfig:
    kind: str = "per_parameter"
    # Feature groups (input ablations): weights | history | adam | metrics | condition
    features: list[str] = field(default_factory=lambda: ["weights", "history", "adam", "metrics", "condition"])
    hidden: list[int] = field(default_factory=lambda: [64, 64])
    coords_per_transition: int = 2048  # stratified equally across tensors
    transitions_per_batch: int = 16
    steps: int = 2000
    eval_every: int = 200
    lr: float = 1e-3
    weight_decay: float = 0.0
    huber_delta: float = 1.0
    target_clip: float = 20.0
    condition_dropout: float = 0.3  # per-objective mask probability during training
    successful_only: bool = False  # ablation: train only on successful transitions (no hindsight)
    # Stage B: behavioural child-loss term (0 disables). Uses training-split minibatches only.
    behavioral_weight: float = 0.0
    behavioral_transitions: int = 4
    behavioral_batch: int = 256
    # Model selection on validation roots: "nde" (parameter-space fit) or "accept_gain" (functional).
    selection: str = "nde"
    functional_val_sources: int = 16
    seed: int = 0


@dataclass
class SafetyConfig:
    max_rel_update: float = 2.0  # per-tensor ||delta|| <= max_rel_update * ||theta|| (blow-up guard only)
    max_abs_value: float = 1e3


@dataclass
class InterleavedConfig:
    """E4: optimizer for ``adam_steps`` -> one improver jump of ``horizon`` -> repeat ``cycles`` times."""

    adam_steps: int = 400  # must be >= the largest history lag
    cycles: int = 6
    horizon: int = 400
    gate: bool = True
    reference_multiple: int = 3  # plain-AdamW reference runs this many times longer (for speedup > 1)
    start_steps: list[int] = field(default_factory=lambda: [400])  # trunk anchor steps to start from
    methods: list[str] = field(
        default_factory=lambda: ["no_update", "linear_extrapolation", "adam_extrapolation", "operator", "operator_scaled"]
    )


@dataclass
class EvaluationConfig:
    root_split: str = "val"  # which root partition to evaluate on (val during development, test once at the end)
    report_split: str = "dev"  # task-data split for reported numbers (test only for the final run)
    horizons: list[int] = field(default_factory=lambda: [100, 400])
    methods: list[str] = field(
        default_factory=lambda: [
            "no_update",
            "linear_extrapolation",
            "adam_extrapolation",
            "history_average",
            "random_norm_matched",
            "foreign_delta",
            "adamw_matched",
            "adamw_full",
            "operator",
            "operator_scaled",
        ]
    )
    request_quantile: float = 0.75
    request_neighbors: int = 50
    # Controllability (H3): sweep the requested loss gain, and request per-class gains.
    request_sweep: list[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])
    class_requests: bool = True
    # Step-size grid for tuned methods; includes 0 so a tuned method may decline to move.
    alpha_grid: list[float] = field(default_factory=lambda: [0.0, 0.01, 0.03, 0.1, 0.25, 0.5, 1.0, 2.0])
    accept_min_gain: float = 0.0
    reference_multiple: int = 4  # reference AdamW curve length = multiple * horizon
    reference_eval_every: int = 10
    reference_smoothing: int = 5  # moving-average window (evaluation points) before the best-so-far envelope
    max_sources_per_root: int | None = None
    bootstrap_samples: int = 2000
    seed: int = 0
    interleaved: InterleavedConfig = field(default_factory=InterleavedConfig)


@dataclass
class ExperimentConfig:
    name: str = "smoke_test"
    output_root: str = "."
    num_threads: int = 1
    task: TaskConfig = field(default_factory=TaskConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    population: PopulationConfig = field(default_factory=PopulationConfig)
    transitions: TransitionConfig = field(default_factory=TransitionConfig)
    operator: OperatorConfig = field(default_factory=OperatorConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    # ---- derived paths -------------------------------------------------
    @property
    def root(self) -> Path:
        return Path(self.output_root)

    @property
    def population_dir(self) -> Path:
        return self.root / "checkpoints" / self.population.name

    @property
    def transitions_path(self) -> Path:
        return self.root / "data" / "transitions" / self.population.name / "transitions.jsonl"

    @property
    def results_dir(self) -> Path:
        return self.root / "results" / self.name

    @property
    def data_dir(self) -> Path:
        p = Path(self.task.data_dir)
        return p if p.is_absolute() else self.root / p

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def section_hash(self, *sections: str) -> str:
        d = self.to_dict()
        payload = {s: d[s] for s in sections}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


class ConfigError(ValueError):
    pass


def _build(cls: type, data: Any, path: str) -> Any:
    if not dataclasses.is_dataclass(cls):
        return data
    if data is None:
        return cls()
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a mapping for {cls.__name__}, got {type(data).__name__}")
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise ConfigError(f"{path}: unknown keys {sorted(unknown)} for {cls.__name__}")
    kwargs = {}
    for key, value in data.items():
        tp = hints[key]
        origin = typing.get_origin(tp)
        args = typing.get_args(tp)
        if dataclasses.is_dataclass(tp):
            kwargs[key] = _build(tp, value, f"{path}.{key}")
        elif origin is list and args and dataclasses.is_dataclass(args[0]):
            kwargs[key] = [_build(args[0], v, f"{path}.{key}[{i}]") for i, v in enumerate(value or [])]
        else:
            kwargs[key] = value
    return cls(**kwargs)


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_yaml_with_inherit(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    parent = data.pop("inherit", None)
    if parent:
        base = _load_yaml_with_inherit((path.parent / parent).resolve())
        data = _deep_merge(base, data)
    return data


def load_config(path: str | Path, overrides: dict | None = None) -> ExperimentConfig:
    data = _load_yaml_with_inherit(Path(path))
    if overrides:
        data = _deep_merge(data, overrides)
    return config_from_dict(data)


def config_from_dict(data: dict) -> ExperimentConfig:
    cfg = _build(ExperimentConfig, data, "config")
    validate_config(cfg)
    return cfg


def validate_config(cfg: ExperimentConfig) -> None:
    s = cfg.population.split
    if abs(s.train + s.val + s.test - 1.0) > 1e-6:
        raise ConfigError("population.split fractions must sum to 1")
    lags = cfg.population.checkpoints.history_lags
    if not lags or any(l <= 0 for l in lags):
        raise ConfigError("population.checkpoints.history_lags must be positive")
    if cfg.population.branching.enabled and not cfg.population.branching.interventions:
        raise ConfigError("branching enabled but no interventions configured")
    known_kinds = {"continue", "lr_scale", "wd_scale", "lr_decay", "class_weight"}
    for iv in cfg.population.branching.interventions:
        if iv.kind not in known_kinds:
            raise ConfigError(f"unknown intervention kind {iv.kind!r}; expected one of {sorted(known_kinds)}")
    known_features = {"weights", "history", "adam", "metrics", "condition"}
    bad = set(cfg.operator.features) - known_features
    if bad:
        raise ConfigError(f"unknown operator feature groups {sorted(bad)}")
    if cfg.operator.selection not in {"nde", "accept_gain"}:
        raise ConfigError("operator.selection must be 'nde' or 'accept_gain'")
    if cfg.evaluation.report_split not in {"dev", "test", "accept"}:
        raise ConfigError("evaluation.report_split must be dev, accept or test")
    if cfg.evaluation.root_split not in {"train", "val", "test"}:
        raise ConfigError("evaluation.root_split must be train, val or test")
