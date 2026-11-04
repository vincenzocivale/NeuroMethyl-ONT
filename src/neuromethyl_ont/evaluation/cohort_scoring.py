"""Shared helpers for scoring real cohorts with the trained linear models.

Checkpoint loading, canonical-parquet feature extraction (binary, continuous
and count features sharing one feature-index column order, so every model
scores exactly the same evidence) and the Consensus ensemble.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from neuromethyl_ont.models.evidence_linear import build_linear_classifier

BINARY_THRESHOLD = 0.6


def load_torch_checkpoint(path: Path):
    """Load a linear-classifier checkpoint; return ``(model, class_names)``."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    classes = [
        label for label, _ in sorted(checkpoint["class_to_idx"].items(), key=lambda p: p[1])
    ]
    model = build_linear_classifier(checkpoint["n_features"], len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, classes


def extract_features(
    files: list[Path], feature_pos: dict[str, int], mu: np.ndarray
) -> dict[str, np.ndarray]:
    """One parquet read per sample -> binary/continuous/count features plus
    per-sample mean coverage, all in feature-index column order."""
    n = len(feature_pos)
    binary = np.zeros((len(files), n), dtype=np.float32)
    continuous = np.zeros((len(files), n), dtype=np.float32)
    m = np.zeros((len(files), n), dtype=np.float64)
    u = np.zeros((len(files), n), dtype=np.float64)
    mean_coverage = np.zeros(len(files), dtype=np.float64)
    for row, path in enumerate(files):
        frame = pd.read_parquet(
            path,
            columns=["probe_id", "coverage", "methylation_fraction", "methylated_count",
                     "unmethylated_count"],
        )
        mean_coverage[row] = frame["coverage"].mean()
        pos = frame["probe_id"].map(feature_pos)
        valid = pos.notna() & (frame["coverage"] > 0) & frame["methylation_fraction"].notna()
        idx = pos.loc[valid].astype(np.int64).to_numpy()
        beta = frame.loc[valid, "methylation_fraction"].to_numpy()
        binary[row, idx] = np.where(beta > BINARY_THRESHOLD, 1.0, -1.0)
        continuous[row, idx] = beta - mu[idx]
        m[row, idx] = frame.loc[valid, "methylated_count"].to_numpy()
        u[row, idx] = frame.loc[valid, "unmethylated_count"].to_numpy()
    return {
        "sample_id": [path.stem for path in files],
        "binary": binary,
        "continuous": continuous,
        "m": m,
        "u": u,
        "mean_coverage": mean_coverage,
    }


def reindex(probs: np.ndarray, source: list[str], target: list[str]) -> np.ndarray:
    """Reorder probability columns from ``source`` to ``target`` label order."""
    out = np.zeros((len(probs), len(target)), dtype=np.float64)
    for col, family in enumerate(source):
        out[:, target.index(family)] = probs[:, col]
    return out


def ensemble_average(*probs: np.ndarray) -> np.ndarray:
    """Arithmetic mean of aligned probability matrices (the Consensus ensemble)."""
    return np.mean(probs, axis=0)
