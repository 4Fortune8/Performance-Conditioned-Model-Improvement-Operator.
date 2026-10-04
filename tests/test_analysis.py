import numpy as np
import pytest

from mio.evaluation.analysis import bootstrap_ci, class_specificity, iqm, summarize


def test_iqm_and_bootstrap():
    x = np.array([0, 1, 2, 3, 4, 5, 6, 100], dtype=float)
    assert iqm(x) == pytest.approx(np.mean([2, 3, 4, 5]))
    lo, hi = bootstrap_ci(np.arange(50, dtype=float), seed=0)
    assert lo < 24.5 < hi


def _row(method, group, source, gain, h=10, stage="mid", valid=True):
    g = {"loss": gain, "acc": 0.0}
    return {
        "method": method, "group_id": group, "source_id": source, "horizon": h, "stage": stage, "valid": valid,
        "accepted": gain > 0, "safety": {}, "gains": {"report": g} if valid else None,
        "gated_gains": {"report": g if gain > 0 else {"loss": 0.0, "acc": 0.0}},
        "equivalent_steps": 1.0, "gap_closed": None, "cos_to_adamw_full": None, "cost": {"fb_equivalent": 0.0},
    }


def test_summarize_paired_by_group():
    rows = []
    for gi, g in enumerate(["a", "b", "c"]):
        for si in range(2):
            s = f"{g}{si}"
            rows.append(_row("no_update", g, s, 0.0))
            rows.append(_row("m", g, s, 0.1 * (gi + 1) + (0.0 if si == 0 else 0.2)))
    out = {r["method"]: r for r in summarize(rows, n_boot=200)}
    m = out["m"]
    assert m["n_groups"] == 3
    assert m["gain_mean"] == pytest.approx(np.mean([0.2, 0.3, 0.4]))  # mean over group means
    assert m["diff_vs_ref_mean"] == pytest.approx(m["gain_mean"])
    assert m["p_group_improves_vs_ref"] == 1.0
    assert out["no_update"]["gain_mean"] == 0.0


def test_class_specificity_diagonal():
    rows = []
    for g in ["a", "b"]:
        for k in range(3):
            gains = {f"class_{j}": (0.5 if j == k else -0.1) for j in range(3)}
            rows.append({"method": "operator_class_request", "valid": True, "requested_class": k, "group_id": g,
                         "gains": {"report": gains}})
    s = class_specificity(rows, 3, n_boot=100)
    assert s["diag_mean"] == pytest.approx(0.5) and s["offdiag_mean"] == pytest.approx(-0.1)
    assert s["specificity_mean"] == pytest.approx(0.6)


def test_interleaved_protocol_calibration(population, synthetic_cfg, tmp_path):
    """no_update must reproduce the plain-AdamW reference exactly (gain 0, speedup 1)."""
    from mio.evaluation.interleaved import run_interleaved
    from mio.pipeline import stage_train_operator
    from mio.utils.serialization import read_jsonl

    stage_train_operator(synthetic_cfg)
    out = run_interleaved(synthetic_cfg, "val", ["no_update", "adam_extrapolation", "operator"], out_dir=tmp_path)
    rows = list(read_jsonl(out / "results.jsonl"))
    base = [r for r in rows if r["method"] == "no_update"]
    assert base and all(r["gain_vs_reference"] == 0.0 for r in base)
    assert all(r["speedup"] == pytest.approx(1.0) for r in base)
    assert {r["method"] for r in rows} == {"no_update", "adam_extrapolation", "operator"}
    assert (out / "summary.md").exists()
