"""Transition index: (source checkpoint -> later checkpoint) records.

Transitions reference checkpoints by id; parameter deltas are computed on load
(``theta[target] - theta[source]``), so horizons and filters can be redefined
without regenerating or duplicating tensors.

Sources are trunk anchors with optimizer state and full lag history. Targets
are (a) later trunk anchors of the same root within ``max_horizon`` and (b)
endpoints of branches spawned *at* the source. Every transition is kept,
successful or not; ``success`` is a label, not a filter. Outcome gains are
recorded per metric split with the sign convention of ``metrics.gains``.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from mio.config import TransitionConfig
from mio.evaluation.metrics import gains
from mio.trajectories.checkpoints import CheckpointStore
from mio.utils.serialization import read_jsonl, write_json, write_jsonl

GAIN_SPLITS = ("train_eval", "accept", "dev")


def build_transitions(store: CheckpointStore, cfg: TransitionConfig) -> list[dict]:
    rows: list[dict] = []
    for src in store.sources():
        for tgt in store.descendants(src["id"]):
            if tgt["kind"] != "anchor":
                continue
            h = tgt["step"] - src["step"]
            if h <= 0 or h > cfg.max_horizon:
                continue
            is_trunk = tgt["line"] == "trunk"
            if is_trunk and not cfg.include_trunk:
                continue
            if not is_trunk:
                if not cfg.include_branches or tgt["branch"]["start_checkpoint"] != src["id"]:
                    continue
            g = {s: gains(src["metrics"][s], tgt["metrics"][s]) for s in GAIN_SPLITS}
            rows.append(
                {
                    "id": f"{src['id']}->{tgt['id']}",
                    "source_id": src["id"],
                    "target_id": tgt["id"],
                    "root_id": src["root_id"],
                    "group_id": src["group_id"],
                    "partition": src["partition"],
                    "source_step": src["step"],
                    "horizon": h,
                    "target_line": tgt["line"],
                    "intervention": tgt["branch"]["intervention"] if not is_trunk else {"name": "trunk", "kind": "trunk"},
                    "source_lr": src["hparams"]["lr"],
                    "gains": g,
                    "success": g[cfg.metric_split]["loss"] > cfg.success_threshold,
                    "cost": {"steps": h, "examples": h * tgt["hparams"]["batch_size"]},
                }
            )
    rows.sort(key=lambda r: (r["root_id"], r["source_step"], r["horizon"], r["target_line"]))
    return rows


def load_transitions(path: str | Path, partition: str | None = None) -> list[dict]:
    return [r for r in read_jsonl(path) if partition in (None, r["partition"])]


def summarize_transitions(rows: list[dict], metric_split: str = "accept") -> dict:
    """Dataset inspection: counts, success rates and gains per intervention / partition."""
    by_iv: dict[str, list[dict]] = defaultdict(list)
    by_part: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_iv[r["intervention"]["name"]].append(r)
        by_part[r["partition"]].append(r)

    def stats(rs: list[dict]) -> dict:
        loss_g = np.array([r["gains"][metric_split]["loss"] for r in rs])
        acc_g = np.array([r["gains"][metric_split]["acc"] for r in rs])
        return {
            "n": len(rs),
            "success_rate": float(np.mean([r["success"] for r in rs])),
            "loss_gain_mean": float(loss_g.mean()),
            "loss_gain_median": float(np.median(loss_g)),
            "acc_gain_mean": float(acc_g.mean()),
            "roots": len({r["root_id"] for r in rs}),
        }

    targeted = [r for r in rows if r["intervention"].get("kind") == "class_weight"]
    class_effect = None
    if targeted:
        on, off = [], []
        for r in targeted:
            k = r["intervention"]["target_class"]
            cg = {key: v for key, v in r["gains"][metric_split].items() if key.startswith("class_")}
            on.append(cg[f"class_{k}"])
            off.extend(v for key, v in cg.items() if key != f"class_{k}")
        class_effect = {
            "n": len(targeted),
            "targeted_class_loss_gain_mean": float(np.mean(on)),
            "other_class_loss_gain_mean": float(np.mean(off)),
        }

    return {
        "metric_split": metric_split,
        "total": stats(rows) if rows else {"n": 0},
        "by_intervention": {k: stats(v) for k, v in sorted(by_iv.items())},
        "by_partition": {k: stats(v) for k, v in sorted(by_part.items())},
        "class_targeting": class_effect,
    }


def write_transitions(rows: list[dict], path: Path, summary: dict) -> None:
    write_jsonl(path, rows)
    write_json(path.with_name("summary.json"), summary)
