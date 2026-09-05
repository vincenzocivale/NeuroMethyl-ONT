#!/usr/bin/env python3
"""
Audit labels, local-file matching, and dataset composition for NeuroMethyl-ONT.

Usage:
    python scripts/data/audit_registry_labels.py /path/to/CNSNanoporeData
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def read_tsv(path):
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_root", type=Path)
    args = ap.parse_args()

    root = args.data_root.resolve()
    registry = root / "registry"
    draft = registry / "drafts" / "geo_samples_draft.tsv"
    audit = registry / "audit"
    audit.mkdir(parents=True, exist_ok=True)

    rows = read_tsv(draft)

    # 1) Sample/file matching summary
    by_ds = defaultdict(list)
    for r in rows:
        by_ds[r["dataset_id"]].append(r)

    summary_rows = []
    for ds, ds_rows in sorted(by_ds.items()):
        n = len(ds_rows)
        matched = sum(bool(r["local_measurement_file"]) for r in ds_rows)
        labels = sum(bool(r["source_label_unharmonized"]) for r in ds_rows)
        summary_rows.append((ds, n, matched, n - matched, labels, n - labels))

    with (audit / "registry_completeness.tsv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["dataset_id", "n_samples", "matched_local_file", "missing_local_file",
                    "has_convenience_label", "missing_convenience_label"])
        w.writerows(summary_rows)

    # 2) Platform/modality + source label distributions
    with (audit / "source_label_counts.tsv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["dataset_id", "modality", "platform_id", "source_label_unharmonized", "count"])
        for ds, ds_rows in sorted(by_ds.items()):
            c = Counter(
                (r["modality"], r["platform_id"], r["source_label_unharmonized"] or "<missing>")
                for r in ds_rows
            )
            for (modality, platform, label), count in c.most_common():
                w.writerow([ds, modality, platform, label, count])

    # 3) All characteristic keys by dataset, to find the actual diagnostic fields.
    with (audit / "characteristic_key_counts.tsv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["dataset_id", "characteristic_key", "samples_with_key", "example_values"])
        for ds, ds_rows in sorted(by_ds.items()):
            key_values = defaultdict(list)
            key_sample_counts = Counter()
            for r in ds_rows:
                chars = json.loads(r["characteristics_json"] or "{}")
                for key, vals in chars.items():
                    key_sample_counts[key] += 1
                    for v in vals:
                        if v not in key_values[key] and len(key_values[key]) < 5:
                            key_values[key].append(v)
            for key in sorted(key_sample_counts):
                w.writerow([
                    ds,
                    key,
                    key_sample_counts[key],
                    " | ".join(key_values[key]),
                ])

    # 4) GSE289246 ONT-only registry draft.
    gse289_ont = [
        r for r in rows
        if r["dataset_id"] == "GSE289246" and r["modality"] == "ONT"
    ]
    if gse289_ont:
        fields = list(gse289_ont[0].keys())
        with (registry / "drafts" / "GSE289246_ONT_only.tsv").open(
            "w", encoding="utf-8", newline=""
        ) as f:
            w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
            w.writeheader()
            w.writerows(gse289_ont)

    # 5) RapidCNS2 file inventory, grouped by subtree.
    rapid_root = root / "datasets" / "RapidCNS2"
    rapid_files = [p for p in rapid_root.rglob("*") if p.is_file() and p.suffix.lower() != ".zip"]
    with (audit / "rapidcns2_files.tsv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["relative_path", "size_bytes", "suffix"])
        for p in sorted(rapid_files):
            w.writerow([p.relative_to(rapid_root), p.stat().st_size, p.suffix.lower() or "<no_ext>"])

    print("Wrote:")
    print(f"  {audit / 'registry_completeness.tsv'}")
    print(f"  {audit / 'source_label_counts.tsv'}")
    print(f"  {audit / 'characteristic_key_counts.tsv'}")
    print(f"  {registry / 'drafts' / 'GSE289246_ONT_only.tsv'}")
    print(f"  {audit / 'rapidcns2_files.tsv'}")
    print()
    print("Registry completeness:")
    for row in summary_rows:
        print("\t".join(map(str, row)))


if __name__ == "__main__":
    main()
