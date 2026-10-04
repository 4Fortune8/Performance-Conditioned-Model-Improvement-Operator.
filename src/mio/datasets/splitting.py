"""Root-level partitioning of a model population.

The partition unit is the *init group*: every root (order variant), anchor,
history snapshot and branch that descends from one initialization is assigned
to the same partition, so no information about a held-out initialization can
reach operator training. Assignment is by a seeded hash order with exact
counts, so it is deterministic, independent of generation order, and never
leaves a partition empty when there are enough groups.
"""

from __future__ import annotations

from typing import Iterable

from mio.config import SplitConfig
from mio.utils.reproducibility import stable_hash

PARTITIONS = ("train", "val", "test")


def assign_partitions(group_ids: Iterable[str], cfg: SplitConfig) -> dict[str, str]:
    groups = sorted(set(group_ids))
    n = len(groups)
    if n == 0:
        return {}
    order = sorted(groups, key=lambda g: stable_hash(f"{cfg.seed}:{g}"))
    n_val = int(round(cfg.val * n))
    n_test = int(round(cfg.test * n))
    if n >= 3:
        n_val = max(n_val, 1) if cfg.val > 0 else 0
        n_test = max(n_test, 1) if cfg.test > 0 else 0
    n_train = n - n_val - n_test
    if n_train < 1:
        raise ValueError(f"not enough init groups ({n}) for a non-empty training partition")
    out: dict[str, str] = {}
    for i, g in enumerate(order):
        out[g] = "train" if i < n_train else ("val" if i < n_train + n_val else "test")
    return out


def check_no_leakage(records: Iterable[dict]) -> None:
    """Every record of one group must be in one partition."""
    seen: dict[str, str] = {}
    for r in records:
        g, p = r["group_id"], r["partition"]
        if seen.setdefault(g, p) != p:
            raise AssertionError(f"group {g} appears in partitions {seen[g]} and {p}")
