"""Update safety, gating/rollback, operator equivariance, training and persistence."""

import dataclasses

import pytest
import torch

from mio.config import SafetyConfig
from mio.evaluation.evaluator import ChildEngine
from mio.evaluation.metrics import objective_names
from mio.evaluation.recursive import RecursionLimits, recursive_improve
from mio.models.base import IncompatibleModelError
from mio.operators.features import FeatureSpec
from mio.operators.residual import ConditionPolicy, CoordinatewiseNet, LearnedOperator
from mio.operators.training import train_operator
from mio.state import Condition, OptState, Proposal


@pytest.fixture(scope="module")
def engine(model, task):
    return ChildEngine(model, task, SafetyConfig(max_rel_update=0.1), report_split="dev")


@pytest.fixture(scope="module")
def src_state(population):
    return population.model_state(population.sources("val")[-1]["id"])


def _parent(engine, state):
    return {"accept": state.metrics, "report": engine.measure(state.theta)["report"]}


def test_nonfinite_rejected_and_rolled_back(engine, src_state):
    delta = torch.zeros_like(src_state.theta)
    delta[0] = float("nan")
    rec = engine.run(src_state, Proposal(delta), _parent(engine, src_state))
    assert not rec["valid"] and not rec["accepted"]
    assert rec["safety"]["rejected"] == "nonfinite_delta"
    assert rec["gated_gains"]["report"]["loss"] == 0.0


def test_structure_mismatch_raises(engine, src_state):
    with pytest.raises(IncompatibleModelError):
        engine.run(src_state, Proposal(torch.zeros(src_state.theta.numel() + 1)), _parent(engine, src_state))


def test_per_tensor_clipping(engine, src_state):
    delta = torch.randn(src_state.theta.numel(), generator=torch.Generator().manual_seed(0)) * 100
    clipped, info = engine.sanitize(src_state.theta, delta)
    assert info["clipped"]
    spec = engine.spec
    ratio = spec.per_tensor_norm(clipped) / spec.per_tensor_norm(src_state.theta)
    assert (ratio <= 0.1 + 1e-5).all()


def test_harmful_update_is_rejected(engine, src_state):
    delta = torch.randn(src_state.theta.numel(), generator=torch.Generator().manual_seed(1))
    rec = engine.run(src_state, Proposal(delta), _parent(engine, src_state))
    assert rec["valid"] and rec["safety"]["clipped"]
    assert rec["gains"]["accept"]["loss"] < 0 and not rec["accepted"]
    assert rec["gated_gains"]["report"]["loss"] == 0.0  # rollback


def _random_operator(spec, lags, seed=0):
    fspec = FeatureSpec(("weights", "history", "adam", "metrics", "condition"), tuple(lags), tuple(objective_names(4)))
    torch.manual_seed(seed)
    net = CoordinatewiseNet(fspec.n_param, fspec.n_global, [16, 16])
    torch.nn.init.normal_(net.body[-1].weight)  # non-trivial output
    gs = {o: 1.0 for o in fspec.objectives}
    return LearnedOperator(net, spec, fspec, gs)


def test_operator_is_permutation_equivariant(model, src_state, population):
    """Permuting hidden units of the parent (theta, history, Adam moments) permutes the proposed delta."""
    op = _random_operator(population.spec, population.meta["history_lags"])
    cond = Condition(50, {"loss": 0.01})
    perm = torch.randperm(16, generator=torch.Generator().manual_seed(4))

    def P(v):
        return model.permute_hidden(v, 0, perm)

    permuted = dataclasses.replace(
        src_state,
        theta=P(src_state.theta),
        history={k: P(v) for k, v in src_state.history.items()},
        opt=OptState(P(src_state.opt.m), P(src_state.opt.v), src_state.opt.step),
    )
    d = op.propose(src_state, cond).delta
    d_perm = op.propose(permuted, cond).delta
    assert d.abs().sum() > 0
    assert torch.allclose(P(d), d_perm, atol=1e-6)


def test_operator_training_and_persistence(population, transitions, synthetic_cfg, tmp_path):
    train = [t for t in transitions if t["partition"] == "train"]
    val = [t for t in transitions if t["partition"] == "val"]
    op, summary = train_operator(synthetic_cfg, population, train, val)
    assert summary["initial_val"]["nde_mean"] == pytest.approx(1.0)  # zero-initialized == no-update
    assert summary["best"]["nde_mean"] < 1.0  # learns something about held-out-root deltas
    path = tmp_path / "op.pt"
    op.save(path, synthetic_cfg.to_dict(), summary)
    loaded = LearnedOperator.load(path, population.spec)
    state = population.model_state(val[0]["source_id"])
    cond = Condition(50, {"loss": 0.02})
    assert torch.equal(op.propose(state, cond).delta, loaded.propose(state, cond).delta)
    other = dataclasses.replace(population.spec, tensors=population.spec.tensors[:-2])
    with pytest.raises(ValueError):
        LearnedOperator.load(path, other)


def test_condition_policy(population, transitions):
    pol = ConditionPolicy([t for t in transitions if t["partition"] == "train"])
    state = population.model_state(population.sources("val")[0]["id"])
    lo = pol.request(state, 50, "loss", 0.1).gains["loss"]
    hi = pol.request(state, 50, "loss", 0.9).gains["loss"]
    assert lo <= hi


def test_recursion_stops(engine, src_state):
    class Fixed:
        name = "fixed"

        def __init__(self, scale):
            self.scale = scale

        def propose(self, state, cond):
            return Proposal(delta=self.scale * torch.ones_like(state.theta))

    parent = _parent(engine, src_state)
    res = recursive_improve(Fixed(0.0), src_state, engine, lambda s: Condition(10), RecursionLimits(max_iters=3), parent)
    assert res.stop_reason == "no further measured improvement" and len(res.steps) == 1
    res = recursive_improve(Fixed(float("nan")), src_state, engine, lambda s: Condition(10), RecursionLimits(), parent)
    assert res.stop_reason.startswith("invalid proposal")
    assert torch.equal(res.final_state.theta, src_state.theta)


def test_behavioral_training_with_functional_selection(population, transitions, synthetic_cfg, task, model):
    train = [t for t in transitions if t["partition"] == "train"]
    val = [t for t in transitions if t["partition"] == "val"]
    cfg = dataclasses.replace(synthetic_cfg, operator=dataclasses.replace(
        synthetic_cfg.operator, behavioral_weight=1.0, behavioral_transitions=2, behavioral_batch=64,
        selection="accept_gain", steps=20, eval_every=10))
    op, summary = train_operator(cfg, population, train, val, task, model)
    assert summary["initial_val"]["accept_gain_mean"] == pytest.approx(0.0, abs=1e-6)  # zero-init == no update
    # functional selection never keeps an operator that is worse than no-update on validation roots
    assert summary["best"]["accept_gain_mean"] >= summary["initial_val"]["accept_gain_mean"]
    assert any(h.get("train_behavioral") is not None for h in summary["history"][1:])
    with pytest.raises(ValueError):
        train_operator(cfg, population, train, val)  # behavioural training needs the task and model
