"""E4: interleaved application (optimizer k steps -> improver jump -> repeat) vs. plain AdamW."""

from _common import config_from_args, parser

from mio.pipeline import stage_interleaved

if __name__ == "__main__":
    p = parser(__doc__)
    p.add_argument("--root-split", choices=["train", "val"], default=None)
    p.add_argument("--methods", nargs="*", default=None)
    args = p.parse_args()
    out = stage_interleaved(config_from_args(args), args.root_split, args.methods)
    print((out / "summary.md").read_text())
