"""Training objectives. V1 uses plain cross-entropy only (docs/EVIDENCE_LINEAR_V1.md #12):
no class weighting in the first run, so this stays a thin, swappable factory
rather than something callers hardcode.
"""

from __future__ import annotations

import torch
from torch import nn


def build_loss(name: str = "cross_entropy", class_weights: torch.Tensor | None = None) -> nn.Module:
    if name == "cross_entropy":
        return nn.CrossEntropyLoss(weight=class_weights)
    raise ValueError(f"unknown loss: {name!r}")


def inverse_sqrt_frequency_weights(class_counts: torch.Tensor) -> torch.Tensor:
    """``1/sqrt(class_frequency)`` weights (spec #12), normalized to mean 1.

    Not used by the first V1 run -- kept here as the documented next step if
    macro-F1 shows a clear rare-class problem, rather than added to the loss
    by default.
    """
    weights = 1.0 / torch.sqrt(class_counts.clamp(min=1).to(torch.float64))
    return (weights / weights.mean()).to(torch.float32)
