"""Population generation, lineage, leakage rules and transition records."""

import pytest
import torch

from mio.datasets.splitting import check_no_leakage
from mio.trajectories.branching import anchor_schedule
from mio.trajectories.checkpoints import CheckpointStore
from mio.trajectories.generator import generate_population


def test_anchor_schedule():
    assert anchor_schedule(300, 25, 2.0, 100) == [0, 25, 50, 100, 200, 300]


def test_manifest_structure(population, synthetic_cfg):
    store = population
    pc = synthetic_cfg.population
    assert len(store.roots()) == pc.n_groups * pc.order_variants
    for rid in store.roots():
        trunk_anchors = [r["step"] for r in store.select(kind="anchor", line="trunk", root_id=rid)]
        assert trunk_anchors == [0, 25, 50, 100, 200, 300]
        for r in store.select(kind="anchor", line="trunk", root_id=rid):
            for lag, hid in r["history"].items():
                assert store.records[hid]["step"] == r["step"] - int(lag)
            assert set(r["metrics"]) == {"train_eval", "accept", "dev"}  # never test
        branches = store.select(kind="anchor", root_id=rid, line="branch")
        assert len(branches) == 3 * pc.branching.n_branches * len(pc.branching.horizons)
        for b in branches:
            start = store.records[b["branch"]["start_checkpoint"]]
            assert b["step"] == start["step"] + b["branch"]["steps_since_branch"]
            assert b["branch"]["intervention"]["name"] in {iv.name for iv in pc.branching.interventions}


def test_no_group_crosses_partitions(population):
    check_no_leakage(population.records.values())
    parts = {r["partition"] for r in population.records.values()}
    assert parts == {"train", "val", "test"}


def test_checkpoint_reload(population):
    store = population
    src = store.sources()[0]
    state = store.model_state(src["id"])
    store.spec.check_vector(state.theta)
    assert state.opt is not None and state.opt.step == src["step"]
    assert set(state.history) == set(store.meta["history_lags"])


def test_regeneration_is_resumable_and_config_locked(population, synthetic_cfg):
    before = len(population)
    generate_population(synthetic_cfg)  # all roots already done -> no retraining
    assert len(CheckpointStore(synthetic_cfg.population_dir)) == before
    import dataclasses
    changed = dataclasses.replace(synthetic_cfg, population=dataclasses.replace(synthetic_cfg.population, total_steps=301))
    with pytest.raises(RuntimeError):
        generate_population(changed)


def test_transitions(transitions, population, synthetic_cfg):
    assert transitions
    for t in transitions:
        src, tgt = population.records[t["source_id"]], population.records[t["target_id"]]
        assert src["root_id"] == tgt["root_id"] and t["partition"] == src["partition"]
        assert 0 < t["horizon"] <= synthetic_cfg.transitions.max_horizon
        g = t["gains"]["accept"]
        assert g["loss"] == pytest.approx(src["metrics"]["accept"]["loss"] - tgt["metrics"]["accept"]["loss"])
        assert t["success"] == (g["loss"] > 0)
    assert {t["success"] for t in transitions} == {True, False}  # unsuccessful transitions are kept
    assert {t["target_line"] == "trunk" for t in transitions} == {True, False}


def test_generation_is_deterministic(synthetic_cfg, population, tmp_path):
    import dataclasses
    from mio.datasets.tasks import load_task
    from mio.trajectories.generator import generate_root, root_list

    task = load_task(synthetic_cfg.task)
    cfg2 = dataclasses.replace(synthetic_cfg, output_root=str(tmp_path))
    root = root_list(cfg2)[0]
    recs = generate_root(cfg2, task, root, tmp_path / "pop")
    last = [r for r in recs if r["line"] == "trunk"][-1]
    from mio.utils.serialization import load_tensors
    from mio.trajectories.checkpoints import checkpoint_path
    regenerated = load_tensors(checkpoint_path(tmp_path / "pop", root["root_id"], last["id"]))["theta"]
    assert torch.equal(regenerated, population.theta(last["id"]))
