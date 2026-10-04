"""Checkpoint storage: safetensors tensors + a JSONL manifest of metadata.

Layout of a population directory::

    population.json            config, ParamSpec, environment, partition map
    manifest.jsonl             one record per checkpoint (assembled from roots/)
    roots/<root_id>.jsonl      per-root records (written when a root completes; enables resume)
    tensors/<root_id>/<checkpoint_id>.safetensors

A checkpoint record (``dict``) has, among others:

    id, root_id, group_id, order_variant, partition
    kind            "anchor" (metrics + optimizer state) | "history" (params only, for lag features)
    line            "trunk" or a branch id
    step            AdamW step count (global, including the trunk before a branch)
    parent_id       previous saved checkpoint on the same line (lineage)
    history         {lag: checkpoint_id} for trunk anchors
    ema_decays      decays of the EMA tensors stored with a trunk anchor (``ema_<decay>``)
    branch          {...} intervention metadata for branch checkpoints
    metrics         {split: metrics} for anchors (train_eval / accept / dev only)
    hparams         optimizer settings in effect
    cost            {"steps_from_init", "examples_seen", "wall_time_s"}
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Iterable

import torch

from mio.models.base import ParamSpec
from mio.state import ModelState, OptState
from mio.utils.serialization import load_tensors, read_json, read_jsonl, save_tensors


def checkpoint_path(population_dir: Path, root_id: str, checkpoint_id: str) -> Path:
    return population_dir / "tensors" / root_id / f"{checkpoint_id}.safetensors"


def ema_key(decay: float) -> str:
    return f"ema_{decay:g}"


def save_checkpoint_tensors(
    population_dir: Path, record: dict, theta: torch.Tensor, opt: OptState | None,
    ema: dict[float, torch.Tensor] | None = None,
) -> None:
    tensors = {"theta": theta}
    if opt is not None:
        tensors["adam_m"] = opt.m
        tensors["adam_v"] = opt.v
    for d, e in (ema or {}).items():
        tensors[ema_key(d)] = e
    save_tensors(
        checkpoint_path(population_dir, record["root_id"], record["id"]),
        tensors,
        metadata={"id": record["id"], "step": str(record["step"]), "adam_step": str(opt.step if opt else -1)},
    )


class CheckpointStore:
    """Read-only access to a generated population."""

    def __init__(self, population_dir: str | Path, cache_size: int = 2048):
        self.dir = Path(population_dir)
        meta_path = self.dir / "population.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"no population at {self.dir} (missing population.json)")
        self.meta = read_json(meta_path)
        self.spec = ParamSpec.from_dict(self.meta["param_spec"])
        self.records: dict[str, dict] = {r["id"]: r for r in read_jsonl(self.dir / "manifest.jsonl")}
        self._cache: OrderedDict[str, dict[str, torch.Tensor]] = OrderedDict()
        self._cache_size = cache_size
        self._children: dict[str, list[str]] = {}
        for r in self.records.values():
            if r.get("parent_id"):
                self._children.setdefault(r["parent_id"], []).append(r["id"])

    # ---- queries ------------------------------------------------------
    def __len__(self) -> int:
        return len(self.records)

    def select(
        self,
        kind: str | None = None,
        partition: str | None = None,
        line: str | None = None,
        root_id: str | None = None,
    ) -> list[dict]:
        out = []
        for r in self.records.values():
            if kind is not None and r["kind"] != kind:
                continue
            if partition is not None and r["partition"] != partition:
                continue
            if line is not None and (r["line"] == "trunk") != (line == "trunk"):
                continue
            if root_id is not None and r["root_id"] != root_id:
                continue
            out.append(r)
        return sorted(out, key=lambda r: (r["root_id"], r["line"], r["step"]))

    def sources(self, partition: str | None = None) -> list[dict]:
        """Trunk anchors usable as operator inputs (have optimizer state and full history)."""
        lags = self.meta["history_lags"]
        return [
            r
            for r in self.select(kind="anchor", partition=partition, line="trunk")
            if r.get("has_optimizer_state") and all(str(l) in r.get("history", {}) for l in lags)
        ]

    def roots(self, partition: str | None = None) -> list[str]:
        return sorted({r["root_id"] for r in self.records.values() if partition in (None, r["partition"])})

    # ---- tensors ------------------------------------------------------
    def _load(self, checkpoint_id: str) -> dict[str, torch.Tensor]:
        if checkpoint_id in self._cache:
            self._cache.move_to_end(checkpoint_id)
            return self._cache[checkpoint_id]
        rec = self.records[checkpoint_id]
        tensors = load_tensors(checkpoint_path(self.dir, rec["root_id"], checkpoint_id))
        self._cache[checkpoint_id] = tensors
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return tensors

    def theta(self, checkpoint_id: str) -> torch.Tensor:
        theta = self._load(checkpoint_id)["theta"]
        self.spec.check_vector(theta)
        return theta

    def opt_state(self, checkpoint_id: str) -> OptState | None:
        rec = self.records[checkpoint_id]
        if not rec.get("has_optimizer_state"):
            return None
        t = self._load(checkpoint_id)
        return OptState(t["adam_m"].clone(), t["adam_v"].clone(), int(rec["step"]))

    def ema(self, checkpoint_id: str) -> dict[float, torch.Tensor]:
        t = self._load(checkpoint_id)
        return {float(d): t[ema_key(float(d))].clone() for d in self.records[checkpoint_id].get("ema_decays", [])}

    def model_state(self, checkpoint_id: str, metric_split: str = "accept") -> ModelState:
        rec = self.records[checkpoint_id]
        history = {int(lag): self.theta(hid) for lag, hid in rec.get("history", {}).items()}
        return ModelState(
            theta=self.theta(checkpoint_id).clone(),
            step=rec["step"],
            history=history,
            opt=self.opt_state(checkpoint_id),
            ema=self.ema(checkpoint_id),
            metrics=rec["metrics"][metric_split],
            hparams=rec["hparams"],
            checkpoint_id=checkpoint_id,
            root_id=rec["root_id"],
            group_id=rec["group_id"],
        )

    def descendants(self, checkpoint_id: str) -> Iterable[dict]:
        """All checkpoints reachable from ``checkpoint_id`` along lineage edges."""
        stack = list(self._children.get(checkpoint_id, []))
        while stack:
            cid = stack.pop()
            yield self.records[cid]
            stack.extend(self._children.get(cid, []))
