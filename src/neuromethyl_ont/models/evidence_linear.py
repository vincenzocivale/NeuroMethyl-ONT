"""The single linear architecture shared by Binary/Beta/Evidence (docs/EVIDENCE_LINEAR_V1.md #5).

Binary, Beta and Evidence differ only in how ``x`` is computed from the raw evidence
(see ``neuromethyl_ont.data.sequencing_simulator``) -- never in model
capacity. This module therefore defines exactly one ``nn.Module``, used
unmodified by all three configs: ``logits = Wx + b``, no hidden layer, no
dropout, no attention. If a richer representation wins under this
architecture, the win cannot be attributed to extra capacity.
"""

from __future__ import annotations

import torch
from torch import nn


class LinearClassifier(nn.Module):
    """``nn.Linear(n_features, n_classes, bias=True)`` and nothing else."""

    def __init__(self, n_features: int, n_classes: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(n_features, n_classes, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def build_linear_classifier(n_features: int, n_classes: int) -> LinearClassifier:
    """Factory used by all three representation configs (Binary/Beta/Evidence) so the
    architecture is enforced by construction rather than duplicated.
    """
    return LinearClassifier(n_features=n_features, n_classes=n_classes)
