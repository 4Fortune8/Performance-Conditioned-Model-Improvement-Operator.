import pytest
import torch

from mio.config import ConfigError, SplitConfig, TaskConfig, config_from_dict, load_config
from mio.datasets.splitting import assign_partitions
from mio.datasets.tasks import ReservedSplitError, load_task


def test_unknown_keys_fail(tmp_path):
    with pytest.raises(ConfigError):
        config_from_dict({"population": {"n_grups": 3}})


def test_inherit_merges(tmp_path):
    (tmp_path / "base.yaml").write_text("name: base\npopulation: {n_groups: 5, total_steps: 100}\n")
    (tmp_path / "child.yaml").write_text("inherit: base.yaml\nname: child\npopulation: {n_groups: 9}\n")
    cfg = load_config(tmp_path / "child.yaml")
    assert cfg.name == "child" and cfg.population.n_groups == 9 and cfg.population.total_steps == 100


def test_split_fractions_validated():
    with pytest.raises(ConfigError):
        config_from_dict({"population": {"split": {"train": 0.5, "val": 0.5, "test": 0.5}}})


def test_partition_assignment_exact_and_deterministic():
    groups = [f"g{i:04d}" for i in range(20)]
    a = assign_partitions(groups, SplitConfig(0.6, 0.2, 0.2, seed=0))
    b = assign_partitions(list(reversed(groups)), SplitConfig(0.6, 0.2, 0.2, seed=0))
    assert a == b
    counts = {p: sum(v == p for v in a.values()) for p in ("train", "val", "test")}
    assert counts == {"train": 12, "val": 4, "test": 4}
    c = assign_partitions(groups, SplitConfig(0.6, 0.2, 0.2, seed=1))
    assert c != a  # the split seed matters


def test_synthetic_task_splits_and_test_guard():
    cfg = TaskConfig(name="synthetic", n_accept=300, n_dev=300, n_train_eval=100)
    t = load_task(cfg)
    assert t.sizes() == {"train": 3400, "train_eval": 100, "accept": 300, "dev": 300, "test": 1000}
    # accept / dev / train are disjoint subsets of the pool
    pool = torch.cat([t.splits[s].x for s in ("train", "accept", "dev")])
    assert torch.unique(pool, dim=0).shape[0] == pool.shape[0]
    with pytest.raises(ReservedSplitError):
        t.split("test")
    assert len(t.split("test", allow_test=True)) == 1000
