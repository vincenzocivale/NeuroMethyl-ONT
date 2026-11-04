"""Two-stage inference: domain router -> CNS or leukemia specialist head.

Each head is the Consensus ensemble (docs/PRODUCTION_MODEL.md) over its own
reference. The router (training/router.py) picks the head from the same sample
counts; if its confidence is below ``threshold`` the sample is left
``undetermined`` rather than forced into either taxonomy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from neuromethyl_ont.data.array_reference import FeatureIndex, load_feature_index
from neuromethyl_ont.data.sequencing_simulator import binary_transform, evidence_transform
from neuromethyl_ont.models.evidence_linear import build_linear_classifier
from neuromethyl_ont.training.router import DOMAINS, predict_domain_proba

UNDETERMINED = "undetermined"


def load_counts_aligned(path: Path, probe_ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Read a probe-level counts TSV and reindex it to ``probe_ids`` order.

    Probes missing from the file get ``m = u = 0`` (unobserved).
    """
    table = pd.read_csv(path, sep="\t")
    required = {"probe_id", "methylated_count", "unmethylated_count"}
    missing_cols = required - set(table.columns)
    if missing_cols:
        raise ValueError(f"{path} is missing columns: {sorted(missing_cols)}")
    if table["probe_id"].duplicated().any():
        raise ValueError(f"{path}: probe_id must be unique (aggregate duplicate rows first)")
    by_probe = table.set_index("probe_id")[["methylated_count", "unmethylated_count"]]
    aligned = by_probe.reindex(probe_ids).fillna(0.0)
    return (
        aligned["methylated_count"].to_numpy(dtype=np.float64),
        aligned["unmethylated_count"].to_numpy(dtype=np.float64),
    )


def load_linear_checkpoint(path: Path, expected_representation: str):
    checkpoint = torch.load(path, map_location="cpu")
    representation = checkpoint["representation"]
    if representation != expected_representation:
        raise ValueError(
            f"{path}: expected representation={expected_representation!r}, "
            f"got {representation!r}"
        )
    class_names = [
        name for name, _ in sorted(checkpoint["class_to_idx"].items(), key=lambda kv: kv[1])
    ]
    model = build_linear_classifier(
        n_features=checkpoint["n_features"], n_classes=len(class_names)
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, class_names


@dataclass
class Head:
    name: str
    feature_index: FeatureIndex
    mu: np.ndarray
    b0_model: nn.Module
    e1_model: nn.Module
    class_names: list[str]
    kappa: float = 2.0


@dataclass
class HeadScores:
    n_observed: int
    x_b0: np.ndarray
    x_e1: np.ndarray
    b0_probs: np.ndarray
    e1_probs: np.ndarray
    ensemble_probs: np.ndarray


def load_head(
    name: str, reference_dir: Path, b0_checkpoint: Path, e1_checkpoint: Path,
    kappa: float = 2.0, feature_index_path: Path | None = None,
) -> Head:
    feature_index = load_feature_index(feature_index_path or reference_dir / "feature_index.json")
    mu = (
        pd.read_parquet(reference_dir / "probe_statistics.parquet")
        .set_index("probe_id")["mean_beta"]
        .reindex(feature_index.probe_ids)
        .to_numpy()
    )
    b0_model, classes = load_linear_checkpoint(b0_checkpoint, "binary")
    e1_model, e1_classes = load_linear_checkpoint(e1_checkpoint, "evidence")
    if e1_classes != classes:
        raise ValueError("Binary and Evidence checkpoints must cover exactly the same classes, in order")
    return Head(name, feature_index, mu, b0_model, e1_model, classes, kappa)


def _softmax(model: nn.Module, x: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        logits = model(torch.as_tensor(x, dtype=torch.float32).unsqueeze(0))[0]
        return torch.softmax(logits, dim=0).numpy()


def score_head(head: Head, m: np.ndarray, u: np.ndarray) -> HeadScores:
    """``m``/``u`` must already be aligned to ``head.feature_index``."""
    x_b0 = binary_transform(m, u)
    x_e1 = evidence_transform(m, u, head.mu, head.kappa)
    b0 = _softmax(head.b0_model, x_b0)
    e1 = _softmax(head.e1_model, x_e1)
    return HeadScores(int((m + u > 0).sum()), x_b0, x_e1, b0, e1, 0.5 * b0 + 0.5 * e1)


def load_router(path: Path) -> tuple[nn.Linear, np.ndarray, str]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    mask = np.asarray(ckpt["shared_probe_mask"], dtype=bool)
    model = nn.Linear(int(mask.size), len(DOMAINS))
    model.load_state_dict(ckpt["model_state_dict"])
    return model.eval(), mask, ckpt["feature_index_checksum"]


def route(
    router: nn.Linear, mask: np.ndarray, m: np.ndarray, u: np.ndarray
) -> np.ndarray:
    """P(domain) for one sample; ``m``/``u`` aligned to the router's (CNS) feature index."""
    x = (binary_transform(m, u) * mask).astype(np.float32)[None, :]
    return predict_domain_proba(router, x)[0]


def decide_domain(proba: np.ndarray, threshold: float) -> str:
    if float(proba.max()) < threshold:
        return UNDETERMINED
    return DOMAINS[int(proba.argmax())]
