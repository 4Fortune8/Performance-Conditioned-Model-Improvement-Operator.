"""End-to-end smoke test: generate -> transitions -> train operator -> evaluate (CPU, minutes)."""

import json
import time

from _common import config_from_args, parser

from mio.pipeline import stage_evaluate, stage_generate, stage_train_operator, stage_transitions

if __name__ == "__main__":
    p = parser(__doc__)
    p.set_defaults(config="configs/smoke_test.yaml")
    for a in p._actions:
        if a.dest == "config":
            a.required = False
    p.add_argument("--workers", type=int, default=None)
    args = p.parse_args()
    cfg = config_from_args(args)
    t0 = time.perf_counter()
    stage_generate(cfg, args.workers)
    print(json.dumps(stage_transitions(cfg)["by_intervention"], indent=1))
    stage_train_operator(cfg)
    out = stage_evaluate(cfg)
    print((out / "summary.md").read_text())
    print(f"smoke test finished in {time.perf_counter() - t0:.1f}s; outputs in {out}")
