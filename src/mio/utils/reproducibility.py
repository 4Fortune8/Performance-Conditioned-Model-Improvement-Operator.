"""Seed derivation and determinism helpers.

All randomness in the pipeline is derived from a single ``master_seed`` through
``derive_seed(master_seed, *keys)``. Keys are strings or ints naming *what* the
seed is for (e.g. ``("init", group_index)``), so adding roots, branches or
experiments never perturbs the seeds of existing ones.
"""

from __future__ import annotations

import hashlib
import os
import random
import subprocess
from pathlib import Path

import numpy as np
import torch


def _key_to_int(key: str | int) -> int:
    if isinstance(key, int):
        if key < 0:
            raise ValueError(f"seed keys must be non-negative, got {key}")
        return key
    digest = hashlib.sha256(str(key).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def derive_seed(master_seed: int, *keys: str | int) -> int:
    """Derive a stable 63-bit seed from a master seed and a path of keys."""
    seq = np.random.SeedSequence(entropy=master_seed, spawn_key=tuple(_key_to_int(k) for k in keys))
    return int(seq.generate_state(1, dtype=np.uint64)[0] >> np.uint64(1))


def stable_hash(text: str) -> int:
    """Process-independent hash (Python's ``hash`` is salted per process)."""
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "little")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def configure_determinism(num_threads: int | None = 1) -> None:
    """Make CPU training bitwise reproducible.

    Single-threaded execution is the simplest way to get bitwise-identical CPU
    results for small models; parallelism is obtained across roots instead
    (one process per root).
    """
    if num_threads is not None:
        torch.set_num_threads(num_threads)
    torch.use_deterministic_algorithms(True, warn_only=True)


def git_commit(cwd: str | Path | None = None) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, check=True, timeout=10
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def environment_info() -> dict:
    return {
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cpu_count": os.cpu_count(),
    }
