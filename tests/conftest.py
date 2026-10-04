from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mio.config import load_config  # noqa: E402


@pytest.fixture(scope="session")
def synthetic_cfg(tmp_path_factory):
    out = tmp_path_factory.mktemp("synthetic")
    return load_config(ROOT / "configs" / "synthetic_test.yaml", {"output_root": str(out)})


@pytest.fixture(scope="session")
def population(synthetic_cfg):
    """A generated synthetic population + transition index (built once per session)."""
    from mio.pipeline import stage_generate, stage_transitions
    from mio.trajectories.checkpoints import CheckpointStore

    stage_generate(synthetic_cfg)
    stage_transitions(synthetic_cfg)
    return CheckpointStore(synthetic_cfg.population_dir)


@pytest.fixture(scope="session")
def task(synthetic_cfg):
    from mio.datasets.tasks import load_task

    return load_task(synthetic_cfg.task, synthetic_cfg.data_dir)


@pytest.fixture(scope="session")
def model(synthetic_cfg, task):
    from mio.models.registry import build_model

    return build_model(synthetic_cfg.model, task.input_dim, task.num_classes)


@pytest.fixture(scope="session")
def transitions(population, synthetic_cfg):
    from mio.datasets.transitions import load_transitions

    return load_transitions(synthetic_cfg.transitions_path)
