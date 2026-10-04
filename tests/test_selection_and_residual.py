"""EMA tracking, the best-simple-baseline selector, residual operators, conditioned behavioural
training, and the merge / memorization-gap analysis."""

import dataclasses

import pytest
import torch

from mio.baselines.averaging import EMAWeights, HistoryAverage, averaging_improver
from mio.baselines.conventional import NoUpdate
from mio.baselines.scaling import WeightScaling
from mio.baselines.selector import SelectBest
from mio.config import SafetyConfig
from mio.evaluation.analysis import generalization_gap
from mio.evaluation.compare import comparison_report, merge_runs
from mio.evaluation.evaluator import ChildEngine
from mio.evaluation.metrics import evaluate, objective_names
from mio.operators.features import FeatureBuilder, FeatureSpec
from mio.operators.residual import CoordinatewiseNet, LearnedOperator
from mio.operators.training import TransitionFeatureCache, train_operator
from mio.state import Condition, OptState
from mio.trajectories.training import EMATracker, TrainSettings, train_steps


@pytest.fixture(scope="module")
def engine(model, task):
    return ChildEngine(model, task, SafetyConfig(max_rel_update=1e9), report_split="dev")


def _trunk_anchors(population, root_id):
    return [r for r in population.select(kind="anchor", line="trunk", root_id=root_id) if r["step"] > 0]


def test_stored_ema_matches_replay_and_resume(population, model, task, synthetic_cfg):
    """Stored EMAs equal a replay from init, and a tracker resumed at one anchor reproduces the next."""
    oc = synthetic_cfg.population.optimizer
    root = population.roots()[0]
    a1, a2 = _trunk_anchors(population, root)[1:3]
    rec0 = population.records[f"{root}-trunk-s000000"]
    settings = TrainSettings.from_config(oc, rec0["trunk_order_seed"])
    train = task.split("train")

    tracker = EMATracker(synthetic_cfg.population.checkpoints.ema_decays, model.num_params)
    seen = {}

    def cb(k, theta, opt):
        tracker.update(theta)
        if k == a1["step"]:
            seen[a1["id"]] = tracker.corrected()

    train_steps(model, population.theta(rec0["id"]), OptState.zeros(model.num_params), train, settings, a2["step"],
                callback=cb)
    seen[a2["id"]] = tracker.corrected()
    for cid in (a1["id"], a2["id"]):
        stored = population.ema(cid)
        assert sorted(stored) == [0.9, 0.99]
        for d, e in stored.items():
            assert torch.allclose(e, seen[cid][d], atol=1e-6)

    resumed = EMATracker.from_corrected(population.ema(a1["id"]), a1["step"])
    train_steps(model, population.theta(a1["id"]), population.opt_state(a1["id"]), train, settings,
                a2["step"] - a1["step"], data_step_start=a1["step"], callback=lambda k, th, o: resumed.update(th))
    for d, e in population.ema(a2["id"]).items():
        assert torch.allclose(e, resumed.corrected()[d], atol=1e-5)


def test_ema_baseline_and_lookup(population):
    state = population.model_state(population.sources("val")[0]["id"])
    p = EMAWeights(0.9).propose(state, Condition(20))
    assert torch.allclose(state.theta + p.delta, state.ema[0.9], atol=1e-7)
    assert averaging_improver("ema_0.99").name == "ema_0.99"
    assert isinstance(averaging_improver("history_average"), HistoryAverage)
    with pytest.raises(ValueError):
        EMAWeights(0.5).propose(state, Condition(20))


def test_selector_picks_best_accept_candidate(population, model, task, engine):
    accept = task.split("accept")

    def score(state, proposal):
        return state.metrics["loss"] - evaluate(model, state.theta + proposal.delta, accept, task.num_classes)["loss"]

    cands = [NoUpdate(), WeightScaling(-0.05), HistoryAverage(), EMAWeights(0.9)]
    sel = SelectBest(cands, score, len(accept), 32)
    for src in population.sources("val")[:4]:
        state = population.model_state(src["id"])
        p = sel.propose(state, Condition(20))
        individual = {c.name: score(state, c.propose(state, Condition(20))) for c in cands}
        assert p.info["chosen"] == max(individual, key=individual.get)
        assert score(state, p) == pytest.approx(max(individual.values()), abs=1e-7)
        assert score(state, p) >= 0.0  # no_update is a candidate, so the gate never rejects the choice
        assert p.cost.task_forward_examples == 4 * len(accept)


def test_residual_operator_starts_at_base_and_persists(population, transitions, synthetic_cfg, tmp_path):
    lags = tuple(population.meta["history_lags"])
    fspec = FeatureSpec(groups=("weights", "history", "adam", "metrics", "condition"), lags=lags,
                        objectives=tuple(objective_names(4)))
    net = CoordinatewiseNet(fspec.n_param, fspec.n_global, [8])
    gscale = {o: 1.0 for o in fspec.objectives}
    op = LearnedOperator(net, population.spec, fspec, gscale, base="history_average")
    state = population.model_state(population.sources("val")[0]["id"])
    cond = Condition(20, {"loss": 0.01})
    assert torch.allclose(op.propose(state, cond).delta, HistoryAverage().propose(state, cond).delta)
    op.save(tmp_path / "op.pt", synthetic_cfg.to_dict(), {})
    loaded = LearnedOperator.load(tmp_path / "op.pt", population.spec)
    assert loaded.base == "history_average"
    assert torch.equal(loaded.propose(state, cond).delta, op.propose(state, cond).delta)

    # residual targets: base + target * scale reconstructs the recorded delta (when unclipped)
    cache = TransitionFeatureCache(population, FeatureBuilder(population.spec, fspec), "accept", 1e9,
                                   base="history_average")
    t = next(t for t in transitions if t["partition"] == "train")
    s = cache.state(t["source_id"])
    rebuilt = cache.target(t) * FeatureBuilder.target_scale(s, t["horizon"]) + cache.base_delta(t["source_id"], t["horizon"])
    assert torch.allclose(rebuilt, cache.delta(t), atol=1e-6)


def test_conditioned_behavioral_training(population, transitions, synthetic_cfg, task, model):
    train = [t for t in transitions if t["partition"] == "train"]
    val = [t for t in transitions if t["partition"] == "val"]
    cfg = dataclasses.replace(synthetic_cfg, operator=dataclasses.replace(
        synthetic_cfg.operator, behavioral_weight=1.0, behavioral_transitions=4, behavioral_batch=64,
        behavioral_objective="conditioned", base="history_average", selection="accept_gain", steps=20,
        eval_every=10))
    op, summary = train_operator(cfg, population, train, val, task, model)
    assert op.base == "history_average" and summary["base"] == "history_average"
    first = summary["history"][0]["val"]
    assert "sweep_spearman" in first and "class_specificity" in first
    assert all(h["train_behavioral"] is not None and h["train_behavioral"] >= 0 for h in summary["history"][1:])
    # the trained (final) operator reads the request
    op.net.load_state_dict(op.final_net_state)
    state = population.model_state(val[0]["source_id"])
    lo = op.propose(state, Condition(20, {"loss": -0.05})).delta
    hi = op.propose(state, Condition(20, {"loss": 0.05})).delta
    assert not torch.allclose(lo, hi)


def _row(method, group, source, gain, h=10):
    g = {"loss": gain, "acc": 0.0, "class_0": 0.0}
    return {"method": method, "group_id": group, "source_id": source, "horizon": h, "stage": "mid", "valid": True,
            "accepted": gain > 0, "safety": {}, "gains": {"report": g},
            "gated_gains": {"report": g if gain > 0 else {"loss": 0.0, "acc": 0.0, "class_0": 0.0}},
            "parent_metrics": {"report": {"class_loss": [1.0]}}, "equivalent_steps": None,
            "reference_max_steps": None, "gap_closed": None, "cos_to_adamw_full": None,
            "cost": {"fb_equivalent": 0.0}, "info": {}}


def test_merge_compare_and_gap():
    def run(op_gain, base_gain=0.01, groups=("a", "b", "c"), offset=0.0):
        rows = []
        for gi, g in enumerate(groups):
            s = f"{g}0"
            rows += [_row("no_update", g, s, 0.0), _row("select_simple", g, s, base_gain),
                     _row("operator", g, s, op_gain + offset + 0.001 * gi)]
        return rows

    runs = {"A": run(0.02), "B": run(0.005)}
    merged, warnings = merge_runs(runs)
    assert not warnings
    assert {r["method"] for r in merged} == {"no_update", "select_simple", "A:operator", "B:operator"}
    md, data = comparison_report(runs, {"A": run(0.05, groups=("x", "y", "z")), "B": run(0.005, groups=("x", "y", "z"))},
                                 n_boot=200)
    vs_sel = {(s["method"], s["horizon"]): s for s in data["summary_vs_select_simple"]}
    assert vs_sel[("A:operator", 10)]["gated_diff_vs_ref_mean"] == pytest.approx(0.011)
    gaps = {g["method"]: g for g in data["generalization_gap"]}
    assert gaps["A:operator"]["gap"] == pytest.approx(0.03)  # memorized on its training roots
    assert gaps["B:operator"]["gap"] == pytest.approx(0.0, abs=1e-12)
    assert "Train roots vs held-out roots" in md

    runs["B"][1] = _row("select_simple", "a", "a0", 0.5)  # a baseline that disagrees between runs
    assert merge_runs(runs)[1]


def test_generalization_gap_unpaired():
    tr = [_row("m", g, f"{g}0", 0.1) for g in "abcd"]
    ho = [_row("m", g, f"{g}0", 0.04) for g in "wxyz"]
    (g,) = generalization_gap(tr, ho, n_boot=100)
    assert g["gap"] == pytest.approx(0.06) and g["n_train_groups"] == 4 and g["n_heldout_groups"] == 4


def test_evaluation_with_selectors(population, synthetic_cfg, tmp_path):
    """select_* rows equal the gated outcome of the candidate they chose."""
    from mio.evaluation.evaluator import run_evaluation
    from mio.pipeline import stage_train_operator
    from mio.utils.serialization import read_jsonl

    if not (synthetic_cfg.results_dir / "operator" / "operator.pt").exists():
        stage_train_operator(synthetic_cfg)
    cfg = dataclasses.replace(synthetic_cfg, evaluation=dataclasses.replace(
        synthetic_cfg.evaluation, class_requests=False, request_sweep=[],
        selector_candidates=["no_update", "history_average", "weight_scaling", "ema_0.9"]))
    methods = ["no_update", "history_average", "weight_scaling", "ema_0.9", "operator", "select_simple",
               "select_with_operator"]
    out = run_evaluation(cfg, "val", "dev", methods=methods, out_dir=tmp_path, with_reference=False)
    rows = list(read_jsonl(out / "results.jsonl"))
    index = {(r["method"], r["source_id"], r["horizon"]): r for r in rows}
    sel = [r for r in rows if r["method"].startswith("select_")]
    assert sel and all(r["equivalent_steps"] is None for r in rows)
    for r in sel:
        chosen = index[(r["info"]["chosen"], r["source_id"], r["horizon"])]
        assert r["gated_gains"]["report"]["loss"] == pytest.approx(chosen["gated_gains"]["report"]["loss"], abs=1e-6)
        assert r["gains"]["accept"]["loss"] >= -1e-9
