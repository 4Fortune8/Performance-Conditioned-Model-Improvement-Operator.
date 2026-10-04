"""Pipeline stages. Each stage reads the previous stage's outputs from disk, so
stages can be run, re-run or replaced independently (see ``scripts/``)."""

from __future__ import annotations

from pathlib import Path

from mio.config import ExperimentConfig
from mio.datasets.tasks import load_task
from mio.models.registry import build_model
from mio.datasets.transitions import build_transitions, load_transitions, summarize_transitions, write_transitions
from mio.evaluation.analysis import (class_specificity, controllability, markdown_report, selector_choices_table,
                                     summarize)
from mio.evaluation.evaluator import run_evaluation
from mio.evaluation.interleaved import run_interleaved
from mio.operators.training import train_operator
from mio.trajectories.checkpoints import CheckpointStore
from mio.trajectories.generator import generate_population
from mio.utils.logging import get_logger
from mio.utils.reproducibility import configure_determinism, git_commit
from mio.utils.serialization import read_json, read_jsonl, write_json

log = get_logger("mio.pipeline")


def stage_generate(cfg: ExperimentConfig, workers: int | None = None) -> Path:
    configure_determinism(cfg.num_threads)
    return generate_population(cfg, workers)


def stage_transitions(cfg: ExperimentConfig) -> dict:
    store = CheckpointStore(cfg.population_dir)
    rows = build_transitions(store, cfg.transitions)
    summary = summarize_transitions(rows, cfg.transitions.metric_split)
    write_transitions(rows, cfg.transitions_path, summary)
    log.info("wrote %d transitions to %s", len(rows), cfg.transitions_path)
    return summary


def stage_train_operator(cfg: ExperimentConfig, out_path: Path | None = None) -> Path:
    configure_determinism(cfg.num_threads)
    store = CheckpointStore(cfg.population_dir)
    rows = load_transitions(cfg.transitions_path)
    train = [r for r in rows if r["partition"] == "train"]
    val = [r for r in rows if r["partition"] == "val"]
    task = load_task(cfg.task, cfg.data_dir)
    model = build_model(cfg.model, task.input_dim, task.num_classes)
    op, summary = train_operator(cfg, store, train, val, task, model)
    out_path = out_path or cfg.results_dir / "operator" / "operator.pt"
    summary["git_commit"] = git_commit()
    op.save(out_path, cfg.to_dict(), summary)
    write_json(out_path.with_name("training_summary.json"), summary)
    # The last training step too, so selection effects can be inspected (not used by default).
    best_state = {k: v.clone() for k, v in op.net.state_dict().items()}
    op.net.load_state_dict(op.final_net_state)
    op.save(out_path.with_name("operator_final.pt"), cfg.to_dict(), {**summary, "selected": "final"})
    op.net.load_state_dict(best_state)
    best = summary["best"]
    log.info("saved operator to %s (selection=%s, best step %s: val nde %.4f, val accept gain %.4f)", out_path,
             cfg.operator.selection, best.get("step"), best.get("nde_mean", float("nan")),
             best.get("accept_gain_mean", float("nan")))
    return out_path


def stage_evaluate(cfg: ExperimentConfig, root_split: str | None = None, report_split: str | None = None,
                   allow_test: bool = False, methods: list[str] | None = None, with_reference: bool = True,
                   out_dir: Path | None = None) -> Path:
    out = run_evaluation(cfg, root_split, report_split, allow_test, methods, out_dir=out_dir,
                         with_reference=with_reference)
    stage_report(cfg, out)
    return out


def stage_interleaved(cfg: ExperimentConfig, root_split: str | None = None, methods: list[str] | None = None) -> Path:
    return run_interleaved(cfg, root_split, methods)


def stage_report(cfg: ExperimentConfig, eval_dir: Path, reference: str = "no_update") -> str:
    rows = list(read_jsonl(eval_dir / "results.jsonl"))
    info = read_json(eval_dir / "run_info.json")
    ec = cfg.evaluation
    num_classes = len(rows[0]["parent_metrics"]["report"]["class_loss"]) if rows else 0
    summary = summarize(rows, reference, ec.bootstrap_samples, ec.seed)
    by_stage = {s: summarize(rows, reference, ec.bootstrap_samples, ec.seed, stage=s) for s in ("early", "mid", "late")}
    ctrl = controllability(rows, ec.bootstrap_samples, ec.seed)
    spec = class_specificity(rows, num_classes, ec.bootstrap_samples, ec.seed)
    md = markdown_report(summary, ctrl, spec, reference,
                         f"{cfg.name}: {info['root_split']} roots, {info['report_split']} split")
    md += "\n## By training stage (gain mean / diff vs ref)\n\n| stage | horizon | method | groups | gain mean | diff vs ref |\n|---|---|---|---|---|---|\n"
    for stage, ss in by_stage.items():
        for s in ss:
            md += (f"| {stage} | {s['horizon']} | {s['method']} | {s['n_groups']} | {s['gain_mean']:.4f} | "
                   f"{s['diff_vs_ref_mean']:.4f} |\n")
    vs_sel = None
    if any(r["method"] == "select_simple" for r in rows):
        vs_sel = summarize(rows, "select_simple", ec.bootstrap_samples, ec.seed)
        md += ("\n## Gated comparison vs the best-simple-baseline selector (`select_simple`)\n\n"
               "Per-group paired differences of *gated* report-split gains.\n\n"
               "| horizon | method | groups | gated gain | gated diff vs select_simple | 95% CI | P(group>sel) |\n"
               "|---|---|---|---|---|---|---|\n")
        for s in sorted(vs_sel, key=lambda s: (s["horizon"], -s["gated_diff_vs_ref_mean"])):
            md += (f"| {s['horizon']} | {s['method']} | {s['n_groups']} | {s['gated_gain_mean']:.4f} | "
                   f"{s['gated_diff_vs_ref_mean']:.4f} | [{s['gated_diff_vs_ref_ci'][0]:.4f}, "
                   f"{s['gated_diff_vs_ref_ci'][1]:.4f}] | {s['p_group_gated_improves_vs_ref']:.2f} |\n")
        md += "\n" + selector_choices_table(rows)
    md += (f"\nTuned baseline step sizes (validation roots, accept split): `{info['alphas']}`; "
           f"AdamW steps matched to one operator application: {info['matched_steps']}.\n")
    (eval_dir / "summary.md").write_text(md, encoding="utf-8")
    write_json(eval_dir / "summary.json", {"summary": summary, "by_stage": by_stage, "controllability": ctrl,
                                           "class_specificity": spec, "vs_select_simple": vs_sel,
                                           "run_info": info})
    return md
