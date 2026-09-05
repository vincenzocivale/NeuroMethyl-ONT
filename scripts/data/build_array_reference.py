#!/usr/bin/env python3
"""Build the canonical GSE90496 feature reference (docs/EVIDENCE_LINEAR_V1.md #3).

    beta matrix + probe ids + labels (GSE90496 training samples only)
        -> build_array_reference()   (drop insufficient/constant/duplicate probes)
        -> save_array_reference()    ($DATA_ROOT/derived/reference/GSE90496/)

This is the ONLY place the canonical B0/C0/E1 feature index is built. Every
downstream cohort (GSE109379, GSE209865, Rapid-CNS2, ONT-WGS) must be aligned
to the resulting ``feature_index.json`` via
``neuromethyl_ont.data.array_reference.align_to_feature_index`` rather than
recomputing its own probe list.

Expects a beta matrix already assembled from GSE90496 IDATs/sesame output
(samples x probes, NaN for missing) plus a matching probe id list and a
sample_id -> label table. Point ``--beta``/``--probe-ids``/``--labels`` at
whatever produced those (this repo does not yet own an IDAT-to-beta step);
by default they are looked up under the docs/DATA_LAYOUT.md convention:

    $NEUROMETHYL_DATA_ROOT/datasets/GSE90496/processed/beta_matrix.npy
    $NEUROMETHYL_DATA_ROOT/datasets/GSE90496/processed/probe_ids.tsv
    $NEUROMETHYL_DATA_ROOT/datasets/GSE90496/processed/labels.tsv

``labels.tsv`` must have columns ``sample_id`` and ``label``, in the same
row order as the beta matrix.

Usage:
    python scripts/data/build_array_reference.py \\
        [--beta PATH] [--probe-ids PATH] [--labels PATH] \\
        [--min-valid-fraction 0.95] [--output-dir PATH]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from neuromethyl_ont.data.array_reference import (  # noqa: E402
    DEFAULT_MIN_VALID_FRACTION,
    build_array_reference,
    save_array_reference,
)
from neuromethyl_ont.data.paths import get_data_root  # noqa: E402

SOURCE_DATASET = "GSE90496"


def load_probe_ids(path: Path) -> list[str]:
    if path.suffix == ".npy":
        return [str(p) for p in np.load(path, allow_pickle=True)]
    return pd.read_csv(path, sep="\t")["probe_id"].astype(str).tolist()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--beta", type=Path, default=None, help="samples x probes .npy beta matrix")
    ap.add_argument("--probe-ids", type=Path, default=None, help=".tsv (column probe_id) or .npy")
    ap.add_argument("--labels", type=Path, default=None, help=".tsv with columns sample_id, label")
    ap.add_argument("--min-valid-fraction", type=float, default=DEFAULT_MIN_VALID_FRACTION)
    ap.add_argument("--output-dir", type=Path, default=None)
    args = ap.parse_args()

    data_root = get_data_root(required=False)
    processed_dir = (data_root / "datasets" / SOURCE_DATASET / "processed") if data_root else None

    beta_path = args.beta or (processed_dir / "beta_matrix.npy" if processed_dir else None)
    probe_ids_path = args.probe_ids or (processed_dir / "probe_ids.tsv" if processed_dir else None)
    labels_path = args.labels or (processed_dir / "labels.tsv" if processed_dir else None)
    output_dir = args.output_dir or (
        data_root / "derived" / "reference" / SOURCE_DATASET if data_root else None
    )

    if beta_path is None or probe_ids_path is None or labels_path is None or output_dir is None:
        print(
            "Could not resolve default paths (NEUROMETHYL_DATA_ROOT not set). "
            "Pass --beta/--probe-ids/--labels/--output-dir explicitly.",
            file=sys.stderr,
        )
        return 2
    for p in (beta_path, probe_ids_path, labels_path):
        if not p.exists():
            print(f"Missing input file: {p}", file=sys.stderr)
            return 2

    beta = np.load(beta_path).astype(np.float64)
    probe_ids = load_probe_ids(probe_ids_path)
    labels_df = pd.read_csv(labels_path, sep="\t")
    sample_ids = labels_df["sample_id"].astype(str).tolist()
    labels = labels_df["label"].astype(str).tolist()

    print(f"Loaded beta matrix: {beta.shape[0]} samples x {beta.shape[1]} probes")
    ref = build_array_reference(
        beta=beta,
        sample_ids=sample_ids,
        probe_ids=probe_ids,
        labels=labels,
        min_valid_fraction=args.min_valid_fraction,
    )
    print(
        f"Kept {len(ref.probe_ids)}/{len(probe_ids)} probes "
        f"(min_valid_fraction={args.min_valid_fraction}); {len(ref.labels)} samples; "
        f"{len(set(ref.labels))} classes"
    )

    feature_index = save_array_reference(ref, output_dir, source_dataset=SOURCE_DATASET)
    print(
        f"Wrote reference to {output_dir} "
        f"(feature_index checksum={feature_index.checksum[:12]}...)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
