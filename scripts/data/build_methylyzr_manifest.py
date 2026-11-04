"""Build registry/methylyzr_cohort_groups.tsv from the minimal MethyLYZR label table (thin wrapper)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from neuromethyl_ont.data.paths import get_data_root
from neuromethyl_ont.evaluation.cohort_groups import build_manifest
from neuromethyl_ont.evaluation.taxonomy_mapping import load_taxonomy

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", type=Path, default=None, help="methylyzr_labels_minimal.tsv")
    ap.add_argument("--taxonomy", type=Path, default=ROOT / "configs/taxonomies/methylyzr_who_level.yaml")
    ap.add_argument("--out", type=Path, default=ROOT / "registry/methylyzr_cohort_groups.tsv")
    args = ap.parse_args()
    labels_path = args.labels or (
        get_data_root() / "datasets/MethyLYZR/metadata/methylyzr_labels_minimal.tsv"
    )
    labels = pd.read_csv(labels_path, sep="\t", dtype=str).fillna("")
    manifest = build_manifest(labels, load_taxonomy(args.taxonomy))
    manifest.to_csv(args.out, sep="\t", index=False)
    print(f"{len(manifest)} libraries, {manifest['split_group'].nunique()} groups -> {args.out}")


if __name__ == "__main__":
    main()
