"""Gate 1: the class-weighted fine-tuning baseline and the steering comparison."""

import dataclasses

import pytest
import torch

from mio.baselines.conventional import ClassWeightedAdamW, ContinuedAdamW
from mio.config import ConfigError, parse_class_baseline
from mio.evaluation.metrics import evaluate
from mio.evaluation.steering import gate1, gate1_markdown
from mio.state import Condition


def test_parse_class_baseline():
    assert parse_class_baseline("class_ft_w4_x2") == (4.0, 2)
    for bad in ("class_ft_4_2", "class_ft_w0_x1", "foo", "class_ft_w4_x0"):
        with pytest.raises(ConfigError):
            parse_class_baseline(bad)


def test_class_weighted_adamw(population, model, task):
    train, dev = task.split("train"), task.split("dev")
    oc = dict(beta1=0.9, beta2=0.999, eps=1e-8)
    state = population.model_state(population.sources("val")[0]["id"])
    cond = Condition(20, {"class_1": 0.1})
    # weight 1 is plain continued AdamW (same data order)
    plain = ContinuedAdamW(model, train, **oc, steps=15).propose(state, cond).delta
    w1 = ClassWeightedAdamW(model, train, **oc, steps=15, weight=1.0, num_classes=task.num_classes)
    assert torch.allclose(w1.propose(state, cond).delta, plain, atol=1e-7)
    # a strong weight helps the requested class more than plain training does
    w = ClassWeightedAdamW(model, train, **oc, steps=15, weight=16.0, num_classes=task.num_classes)
    p = w.propose(state, cond)
    assert p.info["target_class"] == 1 and p.cost.fb_equivalent == 15
    loss1 = lambda d: evaluate(model, state.theta + d, dev, task.num_classes)["class_loss"][1]  # noqa: E731
    assert loss1(p.delta) < loss1(plain)
    with pytest.raises(ValueError):
        w.propose(state, Condition(20, {"loss": 0.1}))


def test_gate1_evaluation(population, synthetic_cfg, tmp_path):
    from mio.evaluation.evaluator import run_evaluation
    from mio.pipeline import stage_train_operator
    from mio.utils.serialization import read_jsonl

    if not (synthetic_cfg.results_dir / "operator" / "operator.pt").exists():
        stage_train_operator(synthetic_cfg)
    cfg = dataclasses.replace(synthetic_cfg, evaluation=dataclasses.replace(
        synthetic_cfg.evaluation, request_sweep=[], class_requests=True,
        class_baselines=["class_ft_w4_x1", "class_ft_w16_x1", "class_ft_w4_x2"]))
    out = run_evaluation(cfg, "val", "dev", methods=["no_update", "operator"], out_dir=tmp_path,
                         with_reference=False)
    rows = list(read_jsonl(out / "results.jsonl"))
    ft = [r for r in rows if r["method"] == "class_ft_w4_x1"]
    hs = sorted({r["horizon"] for r in rows})
    # horizon-free baselines are computed once per (source, class) and reused across horizons
    by = {(r["source_id"], r["requested_class"], r["horizon"]): r for r in ft}
    for (s, k, h), r in by.items():
        assert r["gains"]["report"] == by[(s, k, hs[0])]["gains"]["report"]
    assert all(r["info"]["target_class"] == r["requested_class"] for r in ft)
    res = gate1(rows, 4, 1, n_boot=100)
    assert set(res["bar"]) == {1, 2} and res["bar"][1]["chosen"] in ("class_ft_w4_x1", "class_ft_w16_x1")
    assert res["decision"] in ("GO", "NO-GO", "INCONCLUSIVE")
    assert "Decision" in gate1_markdown(res)


def _row(method, group, source, k, spec, h=10):
    g = {"loss": 0.01, "acc": 0.0, **{f"class_{j}": (spec if j == k else 0.0) for j in range(3)}}
    m = {"class_acc": [0.5] * 3, "class_loss": [1.0] * 3}
    return {"method": method, "group_id": group, "source_id": source, "requested_class": k, "horizon": h,
            "valid": True, "gains": {"report": g, "accept": g}, "child_metrics": {"report": m},
            "parent_metrics": {"report": m}}


def test_gate1_decision_rule():
    def rows(op, ft):
        out = []
        for gi, grp in enumerate("abcd"):
            for k in range(3):
                out += [_row("operator_class_request", grp, f"{grp}0", k, op + 0.001 * gi),
                        _row("class_ft_w4_x1", grp, f"{grp}0", k, ft)]
        return out

    assert gate1(rows(0.03, 0.01), 3, 13, n_boot=200)["decision"] == "GO"
    res = gate1(rows(0.01, 0.03), 3, 13, n_boot=200)
    assert res["decision"] == "NO-GO"
    assert res["operator"]["spec"] == pytest.approx(0.0115)  # mean of 0.010..0.013 over groups
    assert res["bar"][1]["diff_spec"]["mean"] == pytest.approx(0.0115 - 0.03)
