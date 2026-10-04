"""Merge evaluation runs of several operators on the same parents and compare them.

Example (validation roots, dev split)::

    python scripts/compare_operators.py --out results/pilot_comparison.md \
        --run stageB results/fmnist_pilot_stage_b/eval_val_dev results/fmnist_pilot_stage_b/eval_train_dev \
        --run residual results/fmnist_pilot_residual/eval_val_dev results/fmnist_pilot_residual/eval_train_dev

The optional second directory of each run holds the same operator evaluated on *training* roots
(``evaluate_operator.py --root-split train --no-reference``) for the memorization gap.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mio.evaluation.compare import comparison_report  # noqa: E402
from mio.utils.serialization import read_jsonl, write_json  # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", nargs="+", action="append", required=True, metavar=("LABEL", "DIR"),
                   help="LABEL HELDOUT_EVAL_DIR [TRAIN_ROOT_EVAL_DIR]")
    p.add_argument("--out", required=True, help="markdown output path (a .json with the numbers is written next to it)")
    p.add_argument("--title", default="Operator comparison")
    p.add_argument("--bootstrap", type=int, default=2000)
    args = p.parse_args()
    runs, train_runs = {}, {}
    for spec in args.run:
        if len(spec) not in (2, 3):
            p.error("--run takes LABEL HELDOUT_DIR [TRAIN_DIR]")
        runs[spec[0]] = list(read_jsonl(Path(spec[1]) / "results.jsonl"))
        if len(spec) == 3:
            train_runs[spec[0]] = list(read_jsonl(Path(spec[2]) / "results.jsonl"))
    if train_runs and set(train_runs) != set(runs):
        p.error("give a train-root directory for every run or for none")
    md, data = comparison_report(runs, train_runs or None, args.bootstrap, title=args.title)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md + "\n", encoding="utf-8")
    write_json(out.with_suffix(".json"), data)
    print(md)
