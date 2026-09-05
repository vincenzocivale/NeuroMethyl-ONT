"""The canonical GSE90496 probe/feature reference (docs/EVIDENCE_LINEAR_V1.md #3).

One feature index is built exclusively from GSE90496 training data and reused
identically by B0/C0/E1 and by every validation/test cohort (aligned via
:func:`align_to_feature_index`). Probes are dropped only for: insufficient
valid values, zero variance in training, or duplicate probe IDs -- never by
looking at a validation/test set.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_MIN_VALID_FRACTION = 0.95


@dataclass(frozen=True)
class ArrayReference:
    """A canonical (samples x probes) beta matrix plus its feature index.

    ``beta`` is float64, NaN for missing values, shape ``(n_samples, n_probes)``.
    ``probe_mean``/``probe_std``/``probe_valid_fraction`` are computed over
    ``beta`` (i.e. post-filtering) and are what ``mu_i`` in the E1/C0
    transforms comes from.
    """

    beta: np.ndarray
    sample_ids: list[str]
    probe_ids: list[str]
    labels: list[str]
    probe_mean: np.ndarray
    probe_std: np.ndarray
    probe_valid_fraction: np.ndarray


@dataclass(frozen=True)
class FeatureIndex:
    """The frozen probe order + a checksum to detect drift across runs."""

    probe_ids: list[str]
    checksum: str
    source_dataset: str
    created_at: str

    @property
    def n_features(self) -> int:
        return len(self.probe_ids)


def _checksum(probe_ids: list[str]) -> str:
    return hashlib.sha256("\n".join(probe_ids).encode("utf-8")).hexdigest()


def build_array_reference(
    beta: np.ndarray,
    sample_ids: list[str],
    probe_ids: list[str],
    labels: list[str],
    min_valid_fraction: float = DEFAULT_MIN_VALID_FRACTION,
) -> ArrayReference:
    """Filter probes and compute the training-set statistics used as `mu_i`.

    Drops, in this order: probes below ``min_valid_fraction`` non-NaN values,
    probes with zero variance among valid values, and duplicate probe IDs
    (first occurrence kept). Never drops samples.
    """
    beta = np.asarray(beta, dtype=np.float64)
    if beta.shape != (len(sample_ids), len(probe_ids)):
        raise ValueError(
            f"beta shape {beta.shape} does not match "
            f"(len(sample_ids)={len(sample_ids)}, len(probe_ids)={len(probe_ids)})"
        )
    if len(labels) != len(sample_ids):
        raise ValueError("labels must have one entry per sample_id")

    valid = ~np.isnan(beta)
    valid_fraction = valid.mean(axis=0)
    keep = valid_fraction >= min_valid_fraction

    with np.errstate(invalid="ignore"):
        probe_std = np.nanstd(beta, axis=0)
    keep &= probe_std > 0  # constant-in-training probes carry no signal

    seen: set[str] = set()
    is_duplicate = np.zeros(len(probe_ids), dtype=bool)
    for i, probe_id in enumerate(probe_ids):
        if probe_id in seen:
            is_duplicate[i] = True
        else:
            seen.add(probe_id)
    keep &= ~is_duplicate

    beta_f = beta[:, keep]
    probe_ids_f = [p for p, k in zip(probe_ids, keep, strict=True) if k]

    with np.errstate(invalid="ignore"):
        probe_mean_f = np.nanmean(beta_f, axis=0)
        probe_std_f = np.nanstd(beta_f, axis=0)
    probe_valid_fraction_f = (~np.isnan(beta_f)).mean(axis=0)

    return ArrayReference(
        beta=beta_f,
        sample_ids=list(sample_ids),
        probe_ids=probe_ids_f,
        labels=list(labels),
        probe_mean=probe_mean_f,
        probe_std=probe_std_f,
        probe_valid_fraction=probe_valid_fraction_f,
    )


def save_array_reference(
    ref: ArrayReference, output_dir: str | Path, source_dataset: str = "GSE90496"
) -> FeatureIndex:
    """Write the layout from docs/EVIDENCE_LINEAR_V1.md #3 and return the
    ``FeatureIndex`` that must be reused by every downstream model/cohort.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "beta.npy", ref.beta)
    pd.DataFrame({"sample_id": ref.sample_ids}).to_csv(
        output_dir / "sample_ids.tsv", sep="\t", index=False
    )
    pd.DataFrame({"probe_id": ref.probe_ids}).to_csv(
        output_dir / "probe_ids.tsv", sep="\t", index=False
    )
    pd.DataFrame({"sample_id": ref.sample_ids, "label": ref.labels}).to_csv(
        output_dir / "labels.tsv", sep="\t", index=False
    )
    pd.DataFrame(
        {
            "probe_id": ref.probe_ids,
            "mean_beta": ref.probe_mean,
            "std_beta": ref.probe_std,
            "valid_fraction": ref.probe_valid_fraction,
        }
    ).to_parquet(output_dir / "probe_statistics.parquet", index=False)

    feature_index = FeatureIndex(
        probe_ids=ref.probe_ids,
        checksum=_checksum(ref.probe_ids),
        source_dataset=source_dataset,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    (output_dir / "feature_index.json").write_text(
        json.dumps(
            {
                "probe_ids": feature_index.probe_ids,
                "n_features": feature_index.n_features,
                "checksum": feature_index.checksum,
                "source_dataset": feature_index.source_dataset,
                "created_at": feature_index.created_at,
            },
            indent=2,
        )
    )
    return feature_index


def load_array_reference(input_dir: str | Path) -> ArrayReference:
    input_dir = Path(input_dir)
    beta = np.load(input_dir / "beta.npy")
    sample_ids = pd.read_csv(input_dir / "sample_ids.tsv", sep="\t")["sample_id"].tolist()
    probe_ids = pd.read_csv(input_dir / "probe_ids.tsv", sep="\t")["probe_id"].tolist()
    labels_df = pd.read_csv(input_dir / "labels.tsv", sep="\t")
    stats = pd.read_parquet(input_dir / "probe_statistics.parquet")
    return ArrayReference(
        beta=beta,
        sample_ids=sample_ids,
        probe_ids=probe_ids,
        labels=labels_df["label"].tolist(),
        probe_mean=stats["mean_beta"].to_numpy(),
        probe_std=stats["std_beta"].to_numpy(),
        probe_valid_fraction=stats["valid_fraction"].to_numpy(),
    )


def load_feature_index(path: str | Path) -> FeatureIndex:
    payload = json.loads(Path(path).read_text())
    return FeatureIndex(
        probe_ids=payload["probe_ids"],
        checksum=payload["checksum"],
        source_dataset=payload["source_dataset"],
        created_at=payload["created_at"],
    )


def verify_feature_index_checksum(feature_index: FeatureIndex) -> bool:
    return _checksum(feature_index.probe_ids) == feature_index.checksum


def align_to_feature_index(
    beta: np.ndarray, probe_ids: list[str], feature_index: FeatureIndex
) -> np.ndarray:
    """Reindex a (samples x probes) matrix to the training feature index order.

    Probes in ``feature_index`` absent from ``probe_ids`` become an all-NaN
    (missing) column; probes present in ``probe_ids`` but not in
    ``feature_index`` are dropped. This is how GSE109379/test cohorts are
    forced onto the exact GSE90496 feature space (spec #3).
    """
    beta = np.asarray(beta, dtype=np.float64)
    if beta.shape[1] != len(probe_ids):
        raise ValueError("beta.shape[1] must match len(probe_ids)")

    column_index = {probe_id: i for i, probe_id in enumerate(probe_ids)}
    n_samples = beta.shape[0]
    aligned = np.full((n_samples, feature_index.n_features), np.nan, dtype=np.float64)
    for out_col, probe_id in enumerate(feature_index.probe_ids):
        src_col = column_index.get(probe_id)
        if src_col is not None:
            aligned[:, out_col] = beta[:, src_col]
    return aligned
