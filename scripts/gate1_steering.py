"""Gate 1 (EXPERIMENTS.md §6): conditioned-operator steering vs class-weighted fine-tuning.

Runs class requests for the trained operator and the ``class_ft_w<W>_x<M>`` ladder on validation
roots (dev split; no reference curves), checks that the operator rows reproduce the earlier
evaluation if one exists, and applies the pre-registered decision rule.
"""

import dataclasses
import json

from _common import config_from_args, parser

from mio.evaluation.evaluator import run_evaluation
from mio.evaluation.steering import gate1, gate1_markdown
from mio.utils.serialization import read_jsonl, write_json

LADDER = [f"class_ft_w{w}_x{m}" for m in (1, 2, 4, 16) for w in (4, 16)]  # pre-registered

if __name__ == "__main__":
    p = parser(__doc__)
    p.add_argument("--out", default=None, help="output directory (default results/<name>/gate1_val_dev)")
    args = p.parse_args()
    cfg = config_from_args(args)
    ec = cfg.evaluation
    cfg = dataclasses.replace(cfg, evaluation=dataclasses.replace(
        ec, methods=["no_update", "operator"], request_sweep=[], class_requests=True,
        class_baselines=list(ec.class_baselines or LADDER)))
    out = run_evaluation(cfg, "val", "dev", out_dir=args.out or cfg.results_dir / "gate1_val_dev",
                         with_reference=False)
    rows = list(read_jsonl(out / "results.jsonl"))
    info = json.loads((out / "run_info.json").read_text())
    num_classes = len(rows[0]["parent_metrics"]["report"]["class_loss"])

    # reproduction check against the earlier full evaluation of the same operator
    prev = cfg.results_dir / "eval_val_dev" / "results.jsonl"
    repro = None
    if prev.exists():
        key = lambda r: (r["source_id"], r["horizon"], r["requested_class"])  # noqa: E731
        old = {key(r): r["gains"]["report"]["loss"] for r in read_jsonl(prev)
               if r["method"] == "operator_class_request" and r["valid"]}
        new = {key(r): r["gains"]["report"]["loss"] for r in rows
               if r["method"] == "operator_class_request" and r["valid"]}
        common = old.keys() & new.keys()
        repro = {"n_old": len(old), "n_new": len(new), "n_common": len(common),
                 "max_abs_diff": max((abs(old[k] - new[k]) for k in common), default=None)}

    res = gate1(rows, num_classes, info["matched_steps"], n_boot=ec.bootstrap_samples, seed=ec.seed)
    res["reproduction"] = repro
    md = gate1_markdown(res)
    if repro:
        md += (f"\nReproduction of the earlier operator class-request rows: {repro['n_common']} common rows, "
               f"max |Δ overall gain| = {repro['max_abs_diff']:.2e}.\n")
    (out / "gate1.md").write_text(md)
    write_json(out / "gate1.json", res)
    print(md)
