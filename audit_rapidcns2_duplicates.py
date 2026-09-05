#!/usr/bin/env python3
"""
Audit duplicate probe IDs / genomic coordinates in Rapid-CNS2 and ONT-WGS files.

Purpose
-------
Before canonicalization, determine whether repeated rows correspond to:
  - the same IlmnID at the same genomic coordinate,
  - one IlmnID mapped to multiple coordinates,
  - one coordinate mapped to multiple IlmnIDs.

The script is read-only and processes one file at a time.

Usage examples
--------------
python scripts/data/audit_rapidcns2_duplicates.py "$DATA_ROOT" --cohort Rapid_CNS2 --max-files 10
python scripts/data/audit_rapidcns2_duplicates.py "$DATA_ROOT" --cohort all
"""

import argparse
import csv
from pathlib import Path

COHORTS = {
    "ONT_WGS": Path("datasets/RapidCNS2/raw/ONT_WGS/extracted/ONT_WGS"),
    "Rapid_CNS2": Path("datasets/RapidCNS2/raw/Rapid_CNS2/extracted/Rapid-CNS2"),
}


def clean_token(x):
    return x.strip().strip('"').strip("'").lower()


def detect_header(parts):
    toks = [clean_token(x) for x in parts]
    if "coverage" in toks and "methylation_percentage" in toks:
        return {
            "chr": toks.index("chr") if "chr" in toks else 0,
            "start": toks.index("start"),
            "end": toks.index("end"),
            "coverage": toks.index("coverage"),
            "pct": toks.index("methylation_percentage"),
            "probe": toks.index("ilmnid") if "ilmnid" in toks else len(parts) - 1,
        }
    return None


def parse_row(parts, idx):
    chrom = parts[idx["chr"]]
    start = int(float(parts[idx["start"]]))
    end = int(float(parts[idx["end"]]))
    probe = parts[idx["probe"]]
    return probe, (chrom, start, end)


def audit_file(path):
    idx = None
    n_rows = 0

    probe_first_coord = {}
    coord_first_probe = {}

    dup_probe_rows = 0
    dup_coord_rows = 0
    probe_multi_coord_rows = 0
    coord_multi_probe_rows = 0

    examples = []

    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line_no, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line:
                continue
            parts = line.split()

            hdr = detect_header(parts)
            if hdr is not None:
                idx = hdr
                continue

            if idx is None:
                # Known headerless canonical layout:
                # chr start end coverage methylation_percentage IlmnID
                if len(parts) >= 6:
                    idx = {"chr": 0, "start": 1, "end": 2, "coverage": 3, "pct": 4, "probe": 5}
                else:
                    continue

            try:
                probe, coord = parse_row(parts, idx)
            except Exception:
                continue

            n_rows += 1

            if probe in probe_first_coord:
                dup_probe_rows += 1
                if probe_first_coord[probe] != coord:
                    probe_multi_coord_rows += 1
                    if len(examples) < 20:
                        examples.append(
                            ("probe_multi_coord", path.name, line_no, probe,
                             f"{probe_first_coord[probe]} -> {coord}")
                        )
            else:
                probe_first_coord[probe] = coord

            if coord in coord_first_probe:
                dup_coord_rows += 1
                if coord_first_probe[coord] != probe:
                    coord_multi_probe_rows += 1
                    if len(examples) < 20:
                        examples.append(
                            ("coord_multi_probe", path.name, line_no, str(coord),
                             f"{coord_first_probe[coord]} -> {probe}")
                        )
            else:
                coord_first_probe[coord] = probe

    return {
        "sample_file": path.name,
        "n_rows": n_rows,
        "n_unique_probe_ids": len(probe_first_coord),
        "n_unique_coordinates": len(coord_first_probe),
        "duplicate_probe_rows": dup_probe_rows,
        "duplicate_coordinate_rows": dup_coord_rows,
        "probe_multi_coordinate_rows": probe_multi_coord_rows,
        "coordinate_multi_probe_rows": coord_multi_probe_rows,
        "frac_duplicate_probe_rows": dup_probe_rows / n_rows if n_rows else 0.0,
        "frac_duplicate_coordinate_rows": dup_coord_rows / n_rows if n_rows else 0.0,
    }, examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_root", type=Path)
    ap.add_argument("--cohort", choices=["ONT_WGS", "Rapid_CNS2", "all"], default="all")
    ap.add_argument("--max-files", type=int, default=None)
    args = ap.parse_args()

    root = args.data_root.resolve()
    audit_dir = root / "registry" / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    cohorts = list(COHORTS) if args.cohort == "all" else [args.cohort]

    rows = []
    all_examples = []

    for cohort in cohorts:
        d = root / COHORTS[cohort]
        files = sorted(
            [p for p in d.iterdir() if p.is_file() and p.suffix.lower() == ".txt" and not p.name.startswith(".")],
            key=lambda p: p.name,
        )
        if args.max_files is not None:
            files = files[:args.max_files]

        print(f"[{cohort}] {len(files)} files")
        for i, p in enumerate(files, 1):
            stats, examples = audit_file(p)
            stats["cohort"] = cohort
            rows.append(stats)
            all_examples.extend(examples)

            if i == 1 or i % 10 == 0 or i == len(files):
                print(
                    f"  {i:>3}/{len(files)} {p.name}: "
                    f"rows={stats['n_rows']:,}, "
                    f"unique_probe={stats['n_unique_probe_ids']:,}, "
                    f"dup_probe={stats['frac_duplicate_probe_rows']:.3%}, "
                    f"multi_coord={stats['probe_multi_coordinate_rows']:,}"
                )

    out = audit_dir / "rapidcns2_duplicate_audit.tsv"
    fields = [
        "cohort", "sample_file", "n_rows", "n_unique_probe_ids",
        "n_unique_coordinates", "duplicate_probe_rows",
        "duplicate_coordinate_rows", "probe_multi_coordinate_rows",
        "coordinate_multi_probe_rows", "frac_duplicate_probe_rows",
        "frac_duplicate_coordinate_rows",
    ]
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, delimiter="\t")
        w.writeheader()
        w.writerows(rows)

    exout = audit_dir / "rapidcns2_duplicate_examples.tsv"
    with exout.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["kind", "sample_file", "line_number", "key", "detail"])
        w.writerows(all_examples)

    print(f"Wrote: {out}")
    print(f"Wrote: {exout}")


if __name__ == "__main__":
    main()
