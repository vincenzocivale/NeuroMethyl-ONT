#!/usr/bin/env python3
"""Turn a raw GEO series (SOFT family + beta-value matrix) into the standardized
inputs ``build_array_reference.py`` expects (docs/EVIDENCE_LINEAR_V1.md #3):

    $DATA_ROOT/datasets/<GSE_ID>/processed/
        beta_matrix.npy   [n_samples x n_probes] float32
        probe_ids.tsv     column probe_id
        labels.tsv        columns sample_id (GSM), label (methylation class)
        sample_metadata.tsv   + sample_title, sample_column (audit trail)

Written for the GSE90496 / GSE109379 series-matrix format (see
neuromethyl_ont.data.geo_series): one ``*_family.soft.gz`` under
``metadata/`` and one probe x sample ``*.txt.gz`` matrix (beta value +
Detection Pval columns) under ``processed/``. Samples are joined to matrix
columns by the literal ``SAMPLE <n>`` header string, not by position.

Usage:
    python scripts/data/prepare_geo_reference_inputs.py \\
        "$NEUROMETHYL_DATA_ROOT/datasets/GSE90496" \\
        [--soft-family PATH] [--beta-matrix PATH] [--dataset-id GSE90496]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from neuromethyl_ont.data.geo_series import parse_soft_family, read_geo_beta_matrix  # noqa: E402


def find_single(patterns: list[Path]) -> Path:
    matches = [p for pattern_dir, glob in patterns for p in pattern_dir.glob(glob)]
    if not matches:
        searched = ", ".join(f"{d}/{g}" for d, g in patterns)
        raise FileNotFoundError(f"no file found under: {searched}")
    if len(matches) > 1:
        raise FileNotFoundError(f"expected exactly one match, found {len(matches)}: {matches}")
    return matches[0]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("dataset_dir", type=Path, help="e.g. $NEUROMETHYL_DATA_ROOT/datasets/GSE90496")
    ap.add_argument("--soft-family", type=Path, default=None)
    ap.add_argument("--beta-matrix", type=Path, default=None)
    ap.add_argument("--dataset-id", default=None, help="defaults to dataset_dir's basename")
    args = ap.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    dataset_id = args.dataset_id or dataset_dir.name

    soft_family = args.soft_family or find_single(
        [(dataset_dir / "metadata", "*_family.soft.gz")]
    )
    beta_matrix_path = args.beta_matrix or find_single(
        [(dataset_dir / "processed", "**/*.txt.gz")]
    )

    print(f"Parsing labels from {soft_family}")
    labels_df = parse_soft_family(soft_family)
    print(f"Found {len(labels_df)} samples, {labels_df['methylation_class'].nunique()} classes")

    print(f"Reading beta matrix from {beta_matrix_path} ({len(labels_df)} sample columns)")
    beta, probe_ids = read_geo_beta_matrix(
        beta_matrix_path, sample_columns=labels_df["sample_column"].tolist()
    )
    print(f"Beta matrix: {beta.shape[0]} samples x {beta.shape[1]} probes")

    output_dir = dataset_dir / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "beta_matrix.npy", beta)
    pd.DataFrame({"probe_id": probe_ids}).to_csv(
        output_dir / "probe_ids.tsv", sep="\t", index=False
    )
    pd.DataFrame({"sample_id": labels_df["gsm"], "label": labels_df["methylation_class"]}).to_csv(
        output_dir / "labels.tsv", sep="\t", index=False
    )
    labels_df.rename(columns={"gsm": "sample_id"}).to_csv(
        output_dir / "sample_metadata.tsv", sep="\t", index=False
    )

    print(f"Wrote {dataset_id} reference inputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
