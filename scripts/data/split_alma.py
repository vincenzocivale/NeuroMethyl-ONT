#!/usr/bin/env python3
"""Write the patient-disjoint ALMA discovery split.

Output: $NEUROMETHYL_DATA_ROOT/datasets/ALMA/splits/alma_split_v1.tsv with
columns sample_id, patient_id, batch, label, split (train/val/test).
Refuses to overwrite an existing split without --overwrite.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from neuromethyl_ont.data.alma import load_discovery_labels  # noqa: E402
from neuromethyl_ont.data.alma_split import make_patient_split  # noqa: E402
from neuromethyl_ont.data.paths import get_data_root  # noqa: E402

LABEL_COLUMN = "AL Epigenomic Subtype"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    output = args.output or (
        get_data_root() / "datasets" / "ALMA" / "splits" / "alma_split_v1.tsv"
    )
    if output.exists() and not args.overwrite:
        print(f"{output} exists; pass --overwrite to replace it.", file=sys.stderr)
        return 1

    labels = load_discovery_labels()
    labels = labels[labels[LABEL_COLUMN].notna()]
    split = make_patient_split(labels, LABEL_COLUMN, "Patient_ID", seed=args.seed)
    table = labels[["Patient_ID", "Batch", LABEL_COLUMN]].rename(
        columns={"Patient_ID": "patient_id", "Batch": "batch", LABEL_COLUMN: "label"}
    )
    table["split"] = split
    output.parent.mkdir(parents=True, exist_ok=True)
    table.reset_index().to_csv(output, sep="\t", index=False)

    print(f"wrote {output}")
    print(table["split"].value_counts().to_string())
    print(table.groupby("split")["patient_id"].nunique().to_string())
    per_class = table.groupby(["label", "split"]).size().unstack(fill_value=0)
    print(f"min samples per class: {per_class.min().to_dict()}")
    overlap = set(table.loc[table.split == "train", "patient_id"]) & set(
        table.loc[table.split != "train", "patient_id"]
    )
    assert not overlap, "patient leakage"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
