#!/usr/bin/env python3
"""Build the canonical ALMA discovery feature reference (docs/ALMA_MILESTONE.md #3).

    ALMA discovery beta matrix + clinical labels (neuromethyl_ont.data.alma)
        -> drop samples with no `AL Epigenomic Subtype` label
        -> build_array_reference()   (drop insufficient/constant/duplicate probes)
        -> save_array_reference()    ($DATA_ROOT/derived/reference/ALMA_discovery/)

Reuses the same reference machinery as the CNS GSE90496 pipeline
(`neuromethyl_ont.data.array_reference`), but this is a wholly separate
reference: a leukemia classifier built on it must never share a feature
index or class list with Binary/Beta/Evidence.

Usage:
    python scripts/data/build_alma_reference.py [--min-valid-fraction 0.95] [--output-dir PATH]

With ``--split-file`` (see scripts/data/split_alma.py) the reference -- probe
filter and per-probe mean ``mu`` -- is built from the ``train`` rows only, so no
val/test information reaches the features; the val and test samples are
exported, aligned to the train probe order, under
``derived/evaluation/ALMA/{val,test}/`` (beta.npy, probe_ids.tsv, labels.tsv).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from neuromethyl_ont.data.alma import load_discovery_dataset  # noqa: E402
from neuromethyl_ont.data.array_reference import (  # noqa: E402
    DEFAULT_MIN_VALID_FRACTION,
    build_array_reference,
    save_array_reference,
)
from neuromethyl_ont.data.paths import get_data_root  # noqa: E402

SOURCE_DATASET = "ALMA_discovery"
LABEL_COLUMN = "AL Epigenomic Subtype"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--min-valid-fraction", type=float, default=DEFAULT_MIN_VALID_FRACTION)
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--split-file", type=Path, default=None)
    args = ap.parse_args()

    data_root = get_data_root(required=False)
    if data_root is None:
        print("NEUROMETHYL_DATA_ROOT is not set.", file=sys.stderr)
        return 2
    reference_name = SOURCE_DATASET + ("_train" if args.split_file else "")
    output_dir = args.output_dir or (data_root / "derived" / "reference" / reference_name)

    dataset = load_discovery_dataset()
    labeled = dataset[dataset[LABEL_COLUMN].notna()]
    print(
        f"Loaded ALMA discovery dataset: {len(dataset)} samples, "
        f"{labeled[LABEL_COLUMN].nunique()} labeled classes, "
        f"dropping {len(dataset) - len(labeled)} unlabeled samples"
    )

    split_table = None
    if args.split_file:
        split_table = pd.read_csv(args.split_file, sep="\t").set_index("sample_id")
        labeled = labeled.loc[split_table.index]
        print(f"Split file: {split_table['split'].value_counts().to_dict()}")

    non_probe_columns = [c for c in labeled.columns if not c.startswith("cg")]
    probe_ids = [c for c in labeled.columns if c.startswith("cg")]
    all_beta = labeled[probe_ids].to_numpy(dtype="float64")
    all_sample_ids = labeled.index.astype(str).tolist()
    all_labels = labeled[LABEL_COLUMN].astype(str).tolist()
    if split_table is not None:
        train_mask = (split_table.loc[labeled.index, "split"] == "train").to_numpy()
    else:
        train_mask = np.ones(len(labeled), dtype=bool)
    beta = all_beta[train_mask]
    sample_ids = [s for s, k in zip(all_sample_ids, train_mask, strict=True) if k]
    labels = [s for s, k in zip(all_labels, train_mask, strict=True) if k]
    print(f"Beta matrix: {beta.shape[0]} samples x {beta.shape[1]} probes")
    print(f"(dropped non-probe columns: {non_probe_columns})")

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

    if split_table is not None:
        order = [probe_ids.index(p) for p in ref.probe_ids]
        for name in ("val", "test"):
            rows = (split_table.loc[labeled.index, "split"] == name).to_numpy()
            out = data_root / "derived" / "evaluation" / "ALMA" / name
            out.mkdir(parents=True, exist_ok=True)
            np.save(out / "beta.npy", all_beta[rows][:, order])
            pd.DataFrame({"probe_id": ref.probe_ids}).to_csv(
                out / "probe_ids.tsv", sep="\t", index=False
            )
            pd.DataFrame(
                {"sample_id": np.array(all_sample_ids)[rows], "label": np.array(all_labels)[rows]}
            ).to_csv(out / "labels.tsv", sep="\t", index=False)
            print(f"Exported {name}: {int(rows.sum())} samples -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
