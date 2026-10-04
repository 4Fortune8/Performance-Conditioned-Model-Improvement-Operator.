"""Stage 4: compare the operator with baselines on held-out roots and write a report.

Development runs use validation roots and the dev split. The final, one-time
report uses ``--final``, which evaluates on test roots with the official
test split.
"""

from _common import config_from_args, parser

from mio.pipeline import stage_evaluate

if __name__ == "__main__":
    p = parser(__doc__)
    p.add_argument("--root-split", choices=["train", "val", "test"], default=None)
    p.add_argument("--report-split", choices=["dev", "accept", "test"], default=None)
    p.add_argument("--methods", nargs="*", default=None)
    p.add_argument("--final", action="store_true", help="evaluate test roots on the official test split (once)")
    args = p.parse_args()
    cfg = config_from_args(args)
    if args.final:
        out = stage_evaluate(cfg, "test", "test", allow_test=True, methods=args.methods)
    else:
        out = stage_evaluate(cfg, args.root_split, args.report_split, methods=args.methods)
    print((out / "summary.md").read_text())
