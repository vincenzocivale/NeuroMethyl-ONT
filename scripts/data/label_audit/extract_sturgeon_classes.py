#!/usr/bin/env python3
"""Extract Sturgeon's raw + merged operational class list from a model zip.

Migrated from the sibling ``cns-io-genomics-data-audit`` project. Reads
``decoding.json`` (raw class index -> name) and ``merge.json`` (merged-label
-> source-labels) out of a Sturgeon model zip (e.g. ``general.zip``) and
writes two CSVs: the raw decoding and the post-merge operational class list,
each with a normalized ``class_id``, a control-class flag, and an inferred
superfamily. Reinforces docs/EXPERIMENT_POLICY.md's "which classes can a
cohort actually support" question -- run this once per Sturgeon model
release, not per experiment.

Usage:
    python scripts/data/label_audit/extract_sturgeon_classes.py \\
        --model-zip "$NEUROMETHYL_DATA_ROOT/benchmarks/sturgeon/general.zip" \\
        --output outputs/tables/sturgeon_general_classes.csv \\
        --raw-output outputs/tables/sturgeon_general_raw_decoding.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import zipfile
from pathlib import Path
from typing import Any


def normalize_class_id(name: str) -> str:
    value = name.strip().lower()
    value = value.replace("&", "and")
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_")


def is_control_class(name: str) -> bool:
    value = name.lower()
    return (
        value.startswith("control")
        or "control" in value
        or "contr" in value
        or "normal" in value
    )


def infer_superfamily(name: str) -> str:
    value = name.lower()

    if "control" in value or "contr" in value:
        return "control"
    if "embryonal" in value or "mb " in value or "atrt" in value or "etmr" in value:
        return "embryonal_tumor"
    if "ependymal" in value or "epn" in value or "subepn" in value:
        return "ependymal_tumor"
    if "glio-neuronal" in value:
        return "glioneuronal_tumor"
    if "glioblastoma" in value or "gbm" in value or "dmg" in value:
        return "glioblastoma_or_hgg"
    if "glioma idh" in value:
        return "adult_diffuse_glioma_idh"
    if "haematopoeitic" in value or "lympho" in value or "plasma" in value:
        return "haematopoietic_tumor"
    if "melanocytic" in value or "melan" in value:
        return "melanocytic_tumor"
    if "mesenchymal" in value or "chordm" in value or "ews" in value:
        return "mesenchymal_tumor"
    if "meningioma" in value:
        return "meningioma"
    if "other glioma" in value or "lgg" in value or "pa" in value:
        return "other_glioma"
    if "pituitary" in value or "adenopit" in value or "sellar" in value:
        return "sellar_tumor"
    if "schwannoma" in value or "nerve" in value:
        return "nerve_sheath_tumor"

    return "unknown"


def read_json_member(zf: zipfile.ZipFile, member_name: str) -> Any:
    with zf.open(member_name) as f:
        return json.load(f)


def find_member(zf: zipfile.ZipFile, basename: str) -> str:
    matches = [name for name in zf.namelist() if Path(name).name == basename]
    if not matches:
        raise FileNotFoundError(f"Could not find {basename!r} inside model zip.")
    if len(matches) > 1:
        raise RuntimeError(f"Found multiple {basename!r} files inside zip: {matches}")
    return matches[0]


def parse_decoding(decoding: dict[str, str]) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    for key, value in decoding.items():
        try:
            index = int(key)
        except ValueError as exc:
            raise ValueError(f"Expected numeric decoding key, got {key!r}") from exc
        if not isinstance(value, str):
            raise TypeError(f"Expected string class name for decoding key {key!r}")
        rows.append((index, value))
    rows.sort(key=lambda item: item[0])
    return rows


def parse_merge(merge: dict[str, list[str]]) -> dict[str, list[str]]:
    parsed: dict[str, list[str]] = {}
    for merged_label, source_labels in merge.items():
        if not isinstance(merged_label, str):
            raise TypeError("Merge labels must be strings.")
        if not isinstance(source_labels, list):
            raise TypeError(f"Merge value for {merged_label!r} must be a list.")
        if not all(isinstance(label, str) for label in source_labels):
            raise TypeError(f"All source labels for {merged_label!r} must be strings.")
        parsed[merged_label] = source_labels
    return parsed


def build_operational_classes(
    raw_decoding: list[tuple[int, str]], merge_map: dict[str, list[str]]
) -> list[dict[str, str | int | bool]]:
    source_to_merged: dict[str, str] = {}
    for merged_label, source_labels in merge_map.items():
        for source_label in source_labels:
            if source_label in source_to_merged:
                raise ValueError(
                    f"Source label {source_label!r} appears in multiple merge groups: "
                    f"{source_to_merged[source_label]!r} and {merged_label!r}"
                )
            source_to_merged[source_label] = merged_label

    raw_label_set = {label for _, label in raw_decoding}
    missing_sources = sorted(set(source_to_merged) - raw_label_set)
    if missing_sources:
        raise ValueError(
            "Some merge source labels are absent from decoding.json:\n"
            + "\n".join(f"- {label}" for label in missing_sources)
        )

    emitted_merged_labels: set[str] = set()
    operational_rows: list[dict[str, str | int | bool]] = []

    for raw_index, raw_label in raw_decoding:
        merged_label = source_to_merged.get(raw_label)
        if merged_label is None:
            operational_label, was_merged, merged_from = raw_label, False, ""
        else:
            if merged_label in emitted_merged_labels:
                continue
            operational_label = merged_label
            was_merged = True
            merged_from = " | ".join(merge_map[merged_label])
            emitted_merged_labels.add(merged_label)

        operational_rows.append(
            {
                "source_model": "sturgeon_general",
                "class_index": len(operational_rows),
                "raw_first_index": raw_index,
                "raw_class_name": operational_label,
                "class_id": normalize_class_id(operational_label),
                "is_control": is_control_class(operational_label),
                "superfamily": infer_superfamily(operational_label),
                "family": "",
                "operational_class": "",
                "include_in_v0": False,
                "was_merged": was_merged,
                "merged_from": merged_from,
                "curation_status": "raw_extracted",
                "notes": "",
            }
        )

    return operational_rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model-zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-output", type=Path, required=True)
    args = parser.parse_args()

    if not args.model_zip.exists():
        raise FileNotFoundError(f"Missing model zip: {args.model_zip}")

    with zipfile.ZipFile(args.model_zip) as zf:
        decoding = read_json_member(zf, find_member(zf, "decoding.json"))
        merge = read_json_member(zf, find_member(zf, "merge.json"))

    if not isinstance(decoding, dict):
        raise TypeError("decoding.json must be a JSON object.")
    if not isinstance(merge, dict):
        raise TypeError("merge.json must be a JSON object.")

    raw_decoding = parse_decoding(decoding)
    merge_map = parse_merge(merge)
    operational_rows = build_operational_classes(raw_decoding, merge_map)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.raw_output.parent.mkdir(parents=True, exist_ok=True)

    raw_rows = [
        {
            "source_model": "sturgeon_general",
            "raw_index": raw_index,
            "raw_class_name": raw_label,
            "class_id": normalize_class_id(raw_label),
            "is_control": is_control_class(raw_label),
            "superfamily": infer_superfamily(raw_label),
        }
        for raw_index, raw_label in raw_decoding
    ]

    with args.raw_output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(raw_rows[0].keys()))
        writer.writeheader()
        writer.writerows(raw_rows)

    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(operational_rows[0].keys()))
        writer.writeheader()
        writer.writerows(operational_rows)

    print(f"Raw decoding classes: {len(raw_rows)}")
    print(f"Merge groups: {len(merge_map)}")
    print(f"Operational classes after merge: {len(operational_rows)}")
    print(f"Saved raw decoding: {args.raw_output}")
    print(f"Saved operational classes: {args.output}")

    if len(operational_rows) != 87:
        print(
            "WARNING: expected 87 operational classes for Sturgeon General. "
            f"Got {len(operational_rows)}. Inspect decoding.json and merge.json."
        )
    n_controls = sum(bool(row["is_control"]) for row in operational_rows)
    if n_controls != 9:
        print(
            f"WARNING: expected 9 control classes. Auto-detected {n_controls}. "
            "Manual curation may be needed."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
