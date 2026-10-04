"""Shared CLI helpers for the pipeline scripts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mio.config import ExperimentConfig, load_config  # noqa: E402


def parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", required=True, help="experiment YAML (see configs/)")
    p.add_argument("--output-root", default=None, help="override output_root (checkpoints/, data/, results/)")
    return p


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    overrides = {"output_root": args.output_root} if args.output_root else None
    return load_config(args.config, overrides)
