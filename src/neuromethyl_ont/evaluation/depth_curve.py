"""Fixed synthetic depth-curve evaluation (docs/EVIDENCE_LINEAR_V1.md #13/#15).

Builds ONE saved set of synthetic ``(m, u, n)`` realizations per depth on the
validation cohort, then feeds the SAME counts through every representation's
transform so B0/C0/E1 are compared on identical synthetic evidence -- not
independently resampled evidence that happens to have the same lambda.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

from neuromethyl_ont.data.sequencing_simulator import (
    SimulatedCounts,
    apply_representation,
    simulate_counts,
)
from neuromethyl_ont.evaluation.classification import depth_curve_metrics

# Spec #13 validation grid. Distinct from the LAMBDA_GRID training mixture
# (spec #7): this is the fixed set the depth-curve figure is reported over.
DEPTH_GRID: tuple[float, ...] = (0.02, 0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, np.inf)


def depth_label(depth: float) -> str:
    return "full_array" if np.isinf(depth) else str(depth)


def build_fixed_depth_realizations(
    beta: np.ndarray, depth_grid: tuple[float, ...] = DEPTH_GRID, seed: int = 0
) -> dict[float, SimulatedCounts]:
    """One fixed ``SimulatedCounts`` draw per depth, reused across representations.

    A distinct ``rng`` per depth (derived from ``seed`` and the depth's index)
    keeps depths independent while each depth's draw is itself deterministic.
    """
    realizations: dict[float, SimulatedCounts] = {}
    for i, depth in enumerate(depth_grid):
        rng = np.random.default_rng((seed, i))
        realizations[depth] = simulate_counts(beta, float(depth), rng)
    return realizations


def _predict_logits(model: nn.Module, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.as_tensor(x, dtype=torch.float32))
    return logits.numpy()


def run_depth_curve(
    models: dict[str, nn.Module],
    representations: dict[str, str],
    beta: np.ndarray,
    y_true: np.ndarray,
    mu: np.ndarray,
    kappa: float,
    depth_grid: tuple[float, ...] = DEPTH_GRID,
    seed: int = 0,
) -> pd.DataFrame:
    """Evaluate every ``models[name]`` (representation ``representations[name]``)
    at every depth in ``depth_grid``, on shared synthetic counts per depth.

    Returns one row per (model name, depth) with the section-14 metric set.
    """
    realizations = build_fixed_depth_realizations(beta, depth_grid=depth_grid, seed=seed)

    rows = []
    for name, model in models.items():
        representation = representations[name]
        for depth in depth_grid:
            counts = realizations[depth]
            x = apply_representation(representation, beta, counts, mu=mu, kappa=kappa)
            logits = _predict_logits(model, x)
            metrics = depth_curve_metrics(logits, y_true)
            row = {"model": name, "representation": representation, "depth": depth_label(depth)}
            row.update(metrics)
            rows.append(row)

    return pd.DataFrame(rows)
