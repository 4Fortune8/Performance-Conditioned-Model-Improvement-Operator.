"""End-to-end correctness controls and baseline formulas."""

import pytest
import torch

from mio.baselines.averaging import HistoryAverage
from mio.baselines.controls import ForeignDelta, Oracle, RandomNormMatched
from mio.baselines.conventional import ContinuedAdamW, NoUpdate, equivalent_steps, isotonic_nonincreasing
from mio.baselines.extrapolation import AdamExtrapolation, LinearExtrapolation, tune_alpha
from mio.config import SafetyConfig
from mio.evaluation.evaluator import ChildEngine
from mio.evaluation.metrics import evaluate
from mio.state import Condition
from mio.trajectories.training import adam_direction


@pytest.fixture(scope="module")
def engine(model, task):
    return ChildEngine(model, task, SafetyConfig(max_rel_update=1e9), report_split="dev")


def test_oracle_round_trip_reproduces_recorded_metrics(population, transitions, engine):
    """Applying the true recorded delta must reproduce the recorded target metrics."""
    oracle = Oracle(population)
    for t in transitions[:: max(1, len(transitions) // 12)]:
        state = population.model_state(t["source_id"])
        parent = {"accept": state.metrics, "report": population.records[t["source_id"]]["metrics"]["dev"]}
        oracle.target_id = t["target_id"]
        rec = engine.run(state, oracle.propose(state, Condition(t["horizon"])), parent)
        assert rec["valid"]
        for split, recorded in (("accept", "accept"), ("report", "dev")):
            target = population.records[t["target_id"]]["metrics"][recorded]
            assert rec["child_metrics"][split]["loss"] == pytest.approx(target["loss"], abs=1e-5)
            assert rec["child_metrics"][split]["acc"] == pytest.approx(target["acc"], abs=1e-6)
        assert rec["gains"]["accept"]["loss"] == pytest.approx(t["gains"]["accept"]["loss"], abs=1e-5)


def test_continued_adamw_replays_a_recorded_branch(population, model, task, synthetic_cfg):
    """Exact optimizer-state continuation: replaying a 'continue' branch's data order reproduces it."""
    oc = synthetic_cfg.population.optimizer
    branches = [r for r in population.records.values()
                if r["line"] != "trunk" and r["branch"]["intervention"]["kind"] == "continue"]
    assert branches, "synthetic population should contain at least one plain-continue branch"
    b = branches[0]
    state = population.model_state(b["branch"]["start_checkpoint"])
    adamw = ContinuedAdamW(model, task.split("train"), oc.beta1, oc.beta2, oc.eps,
                           steps=b["branch"]["steps_since_branch"], order_seed=b["hparams"]["order_seed"])
    prop = adamw.propose(state, Condition(b["branch"]["steps_since_branch"]))
    # Exact up to float rounding of the delta representation (theta + (theta' - theta)).
    assert torch.allclose(state.theta + prop.delta, population.theta(b["id"]), rtol=0, atol=1e-6)
    assert prop.cost.task_fb_passes == b["branch"]["steps_since_branch"]


def test_baseline_formulas(population, synthetic_cfg):
    oc = synthetic_cfg.population.optimizer
    state = population.model_state(population.sources()[-1]["id"])
    cond = Condition(horizon=40)
    assert torch.count_nonzero(NoUpdate().propose(state, cond).delta) == 0
    lag = max(state.history)
    lin = LinearExtrapolation(lag, alpha=0.5).propose(state, cond).delta
    assert torch.allclose(lin, 0.5 * (40 / lag) * (state.theta - state.history[lag]))
    ad = AdamExtrapolation(oc.beta1, oc.beta2, oc.eps, alpha={40: 2.0}).propose(state, cond).delta
    u = adam_direction(state.opt, oc.beta1, oc.beta2, oc.eps)
    assert torch.allclose(ad, -2.0 * 40 * state.hparams["lr"] * u)
    avg = HistoryAverage().propose(state, cond).delta
    assert torch.allclose(state.theta + avg, torch.stack([state.theta, *state.history.values()]).mean(0), atol=1e-7)
    rnd = RandomNormMatched(population.spec, LinearExtrapolation(lag, 1.0)).propose(state, cond).delta
    ref = LinearExtrapolation(lag, 1.0).propose(state, cond).delta
    assert torch.allclose(population.spec.per_tensor_norm(rnd), population.spec.per_tensor_norm(ref), rtol=1e-4)


def test_foreign_delta_uses_another_root(population, transitions):
    train = [t for t in transitions if t["partition"] == "train"]
    state = population.model_state(population.sources("val")[0]["id"])
    prop = ForeignDelta(population, train).propose(state, Condition(50))
    used = next(t for t in train if t["id"] == prop.info["foreign_transition"])
    assert used["root_id"] != state.root_id


def test_equivalent_steps_uses_isotonic_fit():
    curve = [(0, 1.0), (10, 0.8), (20, 0.9), (30, 0.6)]  # isotonic fit: 1.0, 0.85, 0.85, 0.6
    assert isotonic_nonincreasing([v for _, v in curve]) == pytest.approx([1.0, 0.85, 0.85, 0.6])
    assert equivalent_steps(curve, 1.1) == 0.0
    assert equivalent_steps(curve, 0.9) == pytest.approx(10 * 0.1 / 0.15)
    assert equivalent_steps(curve, 0.7) == pytest.approx(20 + 10 * 0.15 / 0.25)
    assert equivalent_steps(curve, 0.5) is None  # beyond the reference
    # zero gain is zero steps even when early reference points rise above the parent
    assert equivalent_steps([(0, 1.0), (10, 1.2), (20, 0.5)], 1.0) == 0.0


def test_tune_alpha_picks_best(population, model, task):
    states = [population.model_state(r["id"]) for r in population.sources("val")[:2]]
    lag = max(states[0].history)

    def score(s, p):
        return s.metrics["loss"] - evaluate(model, s.theta + p.delta, task.split("accept"), task.num_classes)["loss"]

    best = tune_alpha(lambda a: LinearExtrapolation(lag, a), states, [20], [0.0, 1.0, 100.0], score)
    assert best[20] in (0.0, 1.0)  # a 100x extrapolation should never win
