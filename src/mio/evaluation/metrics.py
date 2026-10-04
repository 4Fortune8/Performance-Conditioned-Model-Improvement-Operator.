"""Model metrics and objective (gain) definitions.

All objectives are expressed as *gains* where positive means better, so
lower-is-better metrics (loss) and higher-is-better metrics (accuracy) share
one sign convention:

    loss      -> parent_loss - child_loss            (NLL, nats)
    acc       -> child_acc - parent_acc
    class_k   -> parent_class_loss[k] - child_class_loss[k]
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from mio.datasets.tasks import Split

ECE_BINS = 15


@torch.no_grad()
def evaluate(model, theta: torch.Tensor, split: Split, num_classes: int, chunk: int = 10000) -> dict:
    nll_parts, correct_parts, conf_parts = [], [], []
    for i in range(0, len(split), chunk):
        x, y = split.x[i : i + chunk], split.y[i : i + chunk]
        logits = model.forward(theta, x)
        nll_parts.append(F.cross_entropy(logits, y, reduction="none"))
        probs = logits.softmax(dim=1)
        conf, pred = probs.max(dim=1)
        correct_parts.append((pred == y).float())
        conf_parts.append(conf)
    nll = torch.cat(nll_parts)
    correct = torch.cat(correct_parts)
    conf = torch.cat(conf_parts)
    y = split.y
    counts = torch.bincount(y, minlength=num_classes).clamp_min(1).float()
    class_loss = torch.zeros(num_classes).index_add_(0, y, nll) / counts
    class_acc = torch.zeros(num_classes).index_add_(0, y, correct) / counts
    return {
        "loss": float(nll.mean()),
        "acc": float(correct.mean()),
        "ece": expected_calibration_error(conf, correct),
        "class_loss": class_loss.tolist(),
        "class_acc": class_acc.tolist(),
        "n": len(split),
    }


def expected_calibration_error(conf: torch.Tensor, correct: torch.Tensor, n_bins: int = ECE_BINS) -> float:
    bins = torch.clamp((conf * n_bins).long(), max=n_bins - 1)
    ece = torch.zeros(())
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            ece += mask.float().mean() * (conf[mask].mean() - correct[mask].mean()).abs()
    return float(ece)


def objective_names(num_classes: int) -> list[str]:
    return ["loss", "acc"] + [f"class_{k}" for k in range(num_classes)]


def gains(parent: dict, child: dict) -> dict[str, float]:
    """Signed improvements of ``child`` over ``parent`` (positive = better)."""
    out = {"loss": parent["loss"] - child["loss"], "acc": child["acc"] - parent["acc"]}
    for k, (pl, cl) in enumerate(zip(parent["class_loss"], child["class_loss"])):
        out[f"class_{k}"] = pl - cl
    return out
