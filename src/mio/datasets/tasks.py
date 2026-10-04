"""Task data and the task-data partition.

The official training set is carved into four fixed, disjoint splits:

    train      - model training (trunks, branches, baseline continuations, behavioural losses)
    train_eval - fixed subset of ``train`` used only to *measure* training metrics
    accept     - metrics fed to operators, outcome labels for conditioning, accept/rollback
    dev        - development reporting (never used for selection inside a run)

The official test set is ``test`` and is only reachable with ``allow_test=True``
so it cannot leak into generation, operator training or candidate selection.
"""

from __future__ import annotations

import gzip
import hashlib
import shutil
import struct
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from mio.config import TaskConfig

_SOURCES = {
    "fashion_mnist": {
        "base": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/",
        "files": {
            "train_x": ("train-images-idx3-ubyte.gz", "8d4fb7e6c68d591d4c3dfef9ec88bf0d"),
            "train_y": ("train-labels-idx1-ubyte.gz", "25c81989df183df01b3e8a0aad5dffbe"),
            "test_x": ("t10k-images-idx3-ubyte.gz", "bef4ecab320f06d8554ea6380940ec79"),
            "test_y": ("t10k-labels-idx1-ubyte.gz", "bb300cfdad3c16e7a12a480ee83cd310"),
        },
    },
    "mnist": {
        "base": "https://ossci-datasets.s3.amazonaws.com/mnist/",
        "files": {
            "train_x": ("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
            "train_y": ("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
            "test_x": ("t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
            "test_y": ("t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c"),
        },
    },
}

SPLITS = ("train", "train_eval", "accept", "dev", "test")


class ReservedSplitError(RuntimeError):
    pass


@dataclass
class Split:
    x: torch.Tensor  # (N, input_dim) float32
    y: torch.Tensor  # (N,) int64

    def __len__(self) -> int:
        return int(self.y.numel())


@dataclass
class TaskData:
    name: str
    input_dim: int
    num_classes: int
    splits: dict[str, Split]

    def split(self, name: str, allow_test: bool = False) -> Split:
        if name == "test" and not allow_test:
            raise ReservedSplitError(
                "the official test split is reserved for the final report; pass allow_test=True explicitly"
            )
        return self.splits[name]

    def sizes(self) -> dict[str, int]:
        return {k: len(v) for k, v in self.splits.items()}


# ---------------------------------------------------------------------------
# IDX download / parsing
# ---------------------------------------------------------------------------


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path, md5: str) -> None:
    if dest.exists() and _md5(dest) == md5:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as resp, open(tmp, "wb") as f:
        shutil.copyfileobj(resp, f)
    got = _md5(tmp)
    if got != md5:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"checksum mismatch for {url}: expected {md5}, got {got}")
    tmp.replace(dest)


def _read_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        data = f.read()
    zero, dtype_code, ndim = struct.unpack(">HBB", data[:4])
    if zero != 0 or dtype_code != 0x08:
        raise ValueError(f"{path}: unsupported IDX header")
    shape = struct.unpack(">" + "I" * ndim, data[4 : 4 + 4 * ndim])
    return np.frombuffer(data, dtype=np.uint8, offset=4 + 4 * ndim).reshape(shape)


def _load_idx_dataset(name: str, data_dir: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    src = _SOURCES[name]
    arrays = {}
    for key, (fname, md5) in src["files"].items():
        dest = data_dir / name / fname
        _download(src["base"] + fname, dest, md5)
        arrays[key] = _read_idx(dest)
    tx = torch.from_numpy(arrays["train_x"].reshape(len(arrays["train_x"]), -1).astype(np.float32) / 255.0)
    ex = torch.from_numpy(arrays["test_x"].reshape(len(arrays["test_x"]), -1).astype(np.float32) / 255.0)
    ty = torch.from_numpy(arrays["train_y"].astype(np.int64))
    ey = torch.from_numpy(arrays["test_y"].astype(np.int64))
    return tx, ty, ex, ey


def _synthetic(cfg: TaskConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    s = cfg.synthetic
    g = torch.Generator().manual_seed(s.seed)
    w1 = torch.randn(s.teacher_hidden, s.input_dim, generator=g) / s.input_dim**0.5
    b1 = torch.randn(s.teacher_hidden, generator=g) * 0.1
    w2 = torch.randn(s.num_classes, s.teacher_hidden, generator=g) / s.teacher_hidden**0.5
    n = s.n_train + s.n_test
    x = torch.randn(n, s.input_dim, generator=g)
    y = (torch.relu(x @ w1.T + b1) @ w2.T).argmax(dim=1)
    return x[: s.n_train], y[: s.n_train], x[s.n_train :], y[s.n_train :]


def load_task(cfg: TaskConfig, data_dir: Path | None = None) -> TaskData:
    data_dir = Path(data_dir if data_dir is not None else cfg.data_dir)
    if cfg.name == "synthetic":
        tx, ty, ex, ey = _synthetic(cfg)
        num_classes = cfg.synthetic.num_classes
        standardize = False
    elif cfg.name in _SOURCES:
        tx, ty, ex, ey = _load_idx_dataset(cfg.name, data_dir)
        num_classes = 10
        standardize = True
    else:
        raise KeyError(f"unknown task {cfg.name!r}")

    n_pool = len(ty)
    if cfg.n_accept + cfg.n_dev >= n_pool:
        raise ValueError("n_accept + n_dev must leave examples for training")
    perm = torch.randperm(n_pool, generator=torch.Generator().manual_seed(cfg.split_seed))
    accept_idx = perm[: cfg.n_accept]
    dev_idx = perm[cfg.n_accept : cfg.n_accept + cfg.n_dev]
    train_idx = perm[cfg.n_accept + cfg.n_dev :]
    if cfg.max_train is not None:
        train_idx = train_idx[: cfg.max_train]
    train_eval_idx = train_idx[: min(cfg.n_train_eval, len(train_idx))]

    if standardize:
        # Statistics from the training split only.
        mean = tx[train_idx].mean()
        std = tx[train_idx].std()
        tx = (tx - mean) / std
        ex = (ex - mean) / std

    splits = {
        "train": Split(tx[train_idx].contiguous(), ty[train_idx].contiguous()),
        "train_eval": Split(tx[train_eval_idx].contiguous(), ty[train_eval_idx].contiguous()),
        "accept": Split(tx[accept_idx].contiguous(), ty[accept_idx].contiguous()),
        "dev": Split(tx[dev_idx].contiguous(), ty[dev_idx].contiguous()),
        "test": Split(ex.contiguous(), ey.contiguous()),
    }
    return TaskData(cfg.name, int(tx.shape[1]), num_classes, splits)
