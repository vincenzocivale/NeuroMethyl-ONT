#!/usr/bin/env python3
"""
Build a provenance-preserving draft sample registry from GEO SOFT files.

This script intentionally does NOT harmonize diagnostic labels.
It preserves GEO metadata as submitted and only adds conservative
technical annotations (dataset role, broad modality, matched local file).

Usage:
    python scripts/data/build_geo_registry_draft.py /path/to/CNSNanoporeData
"""

import argparse
import csv
import gzip
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

DATASETS = {
    "GSE90496": {
        "soft": "datasets/GSE90496/metadata/GSE90496_family.soft.gz",
        "role": "train",
        "default_modality": "array",
    },
    "GSE109379": {
        "soft": "datasets/GSE109379/metadata/GSE109379_family.soft.gz",
        "role": "validation",
        "default_modality": "array",
    },
    "GSE289246": {
        "soft": "datasets/GSE289246/metadata/GSE289246_family.soft.gz",
        "role": "test_primary",
        "default_modality": "",
    },
    "GSE209865": {
        "soft": "datasets/GSE209865/metadata/GSE209865_family.soft.gz",
        "role": "benchmark_legacy",
        "default_modality": "ONT",
    },
}

PLATFORM_MAP = {
    "GPL24106": ("ONT", "MinION"),
    "GPL26167": ("ONT", "PromethION"),
    "GPL24676": ("WGBS", "Illumina NovaSeq 6000"),
}

KEY_NORMALIZE = re.compile(r"[^a-z0-9]+")


def norm_key(s):
    return KEY_NORMALIZE.sub("_", s.strip().lower()).strip("_")


def open_text(path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("rt", encoding="utf-8", errors="replace")


def parse_soft_samples(path):
    samples = []
    current = None

    with open_text(path) as fh:
        for raw in fh:
            line = raw.rstrip("\n\r")

            if line.startswith("^SAMPLE = "):
                if current is not None:
                    samples.append(current)
                current = {
                    "gsm_id": line.split("=", 1)[1].strip(),
                    "characteristics": defaultdict(list),
                    "supplementary_files": [],
                }
                continue

            if current is None:
                continue

            if line.startswith("!Sample_title = "):
                current["title"] = line.split("=", 1)[1].strip()
            elif line.startswith("!Sample_geo_accession = "):
                current["geo_accession"] = line.split("=", 1)[1].strip()
            elif line.startswith("!Sample_platform_id = "):
                current["platform_id"] = line.split("=", 1)[1].strip()
            elif line.startswith("!Sample_source_name_ch1 = "):
                current["source_name"] = line.split("=", 1)[1].strip()
            elif line.startswith("!Sample_organism_ch1 = "):
                current["organism"] = line.split("=", 1)[1].strip()
            elif line.startswith("!Sample_library_strategy = "):
                current["library_strategy"] = line.split("=", 1)[1].strip()
            elif line.startswith("!Sample_description = "):
                current.setdefault("descriptions", []).append(
                    line.split("=", 1)[1].strip()
                )
            elif line.startswith("!Sample_characteristics_ch1 = "):
                payload = line.split("=", 1)[1].strip()
                if ":" in payload:
                    k, v = payload.split(":", 1)
                    current["characteristics"][norm_key(k)].append(v.strip())
                else:
                    current["characteristics"]["unparsed"].append(payload)
            elif line.startswith("!Sample_supplementary_file"):
                current["supplementary_files"].append(
                    line.split("=", 1)[1].strip()
                )

    if current is not None:
        samples.append(current)

    return samples


def all_local_files(data_root, dataset_id):
    root = data_root / "datasets" / dataset_id
    return [p for p in root.rglob("*") if p.is_file()]


def match_local_file(gsm_id, files):
    matches = [
        p for p in files
        if gsm_id.lower() in p.name.lower()
        and ".soft" not in p.name.lower()
    ]
    if not matches:
        return ""

    matches.sort(
        key=lambda p: (
            0 if "extracted" in p.parts else 1,
            len(str(p)),
        )
    )
    return str(matches[0])


def first_matching_characteristic(chars, candidates):
    for cand in candidates:
        key = norm_key(cand)
        vals = chars.get(key)
        if vals:
            return " | ".join(vals)
    return ""


def infer_modality(platform_id, default_modality):
    if platform_id in PLATFORM_MAP:
        return PLATFORM_MAP[platform_id]
    if default_modality == "array":
        return ("array", platform_id)
    if default_modality == "ONT":
        return ("ONT", platform_id)
    return ("unknown", platform_id)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_root", type=Path)
    args = ap.parse_args()

    data_root = args.data_root.resolve()
    registry = data_root / "registry"
    drafts = registry / "drafts"
    audit = registry / "audit"
    drafts.mkdir(parents=True, exist_ok=True)
    audit.mkdir(parents=True, exist_ok=True)

    rows = []
    summary = []

    for dataset_id, cfg in DATASETS.items():
        soft = data_root / cfg["soft"]
        if not soft.exists():
            raise FileNotFoundError(f"Missing SOFT file: {soft}")

        samples = parse_soft_samples(soft)
        files = all_local_files(data_root, dataset_id)
        modality_counts = Counter()
        platform_counts = Counter()

        for s in samples:
            gsm = s.get("geo_accession") or s.get("gsm_id", "")
            platform_id = s.get("platform_id", "")
            modality, platform_name = infer_modality(
                platform_id, cfg["default_modality"]
            )
            modality_counts[modality] += 1
            platform_counts[platform_id or "<missing>"] += 1

            chars = dict(s["characteristics"])
            source_label = first_matching_characteristic(
                chars,
                [
                    "methylation class",
                    "methylation_class",
                    "methylation class family",
                    "methylation_class_family",
                    "diagnosis",
                    "integrated diagnosis",
                    "tumor type",
                    "tumour type",
                    "class",
                ],
            )

            rows.append(
                {
                    "dataset_id": dataset_id,
                    "role": cfg["role"],
                    "gsm_id": gsm,
                    "source_title": s.get("title", ""),
                    "source_name": s.get("source_name", ""),
                    "platform_id": platform_id,
                    "modality": modality,
                    "platform_name": platform_name,
                    "library_strategy": s.get("library_strategy", ""),
                    "source_label_unharmonized": source_label,
                    "local_measurement_file": match_local_file(gsm, files),
                    "supplementary_files_json": json.dumps(
                        s.get("supplementary_files", []), ensure_ascii=False
                    ),
                    "characteristics_json": json.dumps(
                        chars, ensure_ascii=False, sort_keys=True
                    ),
                    "description_json": json.dumps(
                        s.get("descriptions", []), ensure_ascii=False
                    ),
                }
            )

        summary.append(
            (
                dataset_id,
                len(samples),
                "; ".join(f"{k}:{v}" for k, v in sorted(modality_counts.items())),
                "; ".join(f"{k}:{v}" for k, v in sorted(platform_counts.items())),
            )
        )

    out = drafts / "geo_samples_draft.tsv"
    fields = list(rows[0].keys()) if rows else []

    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    summary_out = audit / "geo_sample_summary.tsv"
    with summary_out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(
            ["dataset_id", "n_samples", "modality_counts", "platform_counts"]
        )
        writer.writerows(summary)

    print(f"Wrote: {out}")
    print(f"Wrote: {summary_out}")
    print()
    print("Dataset summary:")
    for row in summary:
        print("\t".join(map(str, row)))
    print()
    print("IMPORTANT: source_label_unharmonized is only a convenience field.")
    print("Use characteristics_json as the provenance-preserving source of truth.")


if __name__ == "__main__":
    main()
