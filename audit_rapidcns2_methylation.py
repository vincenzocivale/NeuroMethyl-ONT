#!/usr/bin/env python3
"""
Audit Rapid-CNS2 / ONT-WGS methylation files.

The script is intentionally read-only. It streams the original text files and
writes QC summaries under DATA_ROOT/registry/audit without modifying raw data.

Outputs
-------
rapidcns2_methylation_qc.tsv
    One row per sample/file with locus count, coverage distribution, and
    consistency of methylation_percentage with integer methylated-read counts.

rapidcns2_probe_space.tsv
    Cohort-level union/intersection statistics for Illumina probe IDs.

rapidcns2_probe_overlap.tsv
    Pairwise overlap summary between the ONT_WGS and Rapid_CNS2 cohort-level
    probe unions/intersections.

Usage
-----
python scripts/data/audit_rapidcns2_methylation.py /path/to/CNSNanoporeData
"""

import argparse
import csv
import math
from pathlib import Path

COHORTS = {
    "ONT_WGS": "datasets/RapidCNS2/raw/ONT_WGS/extracted/ONT_WGS",
    "Rapid_CNS2": "datasets/RapidCNS2/raw/Rapid_CNS2/extracted/Rapid-CNS2",
}


def is_header(parts):
    if not parts:
        return False
    return parts[0].lower() in {"chr", "chrom", "chromosome"} and "coverage" in [p.lower() for p in parts]


def parse_file(path):
    n_loci = 0
    total_cov = 0
    cov_gt1 = cov_ge2 = cov_ge5 = cov_ge10 = cov_ge20 = 0
    methylated_sum = 0
    unmethylated_sum = 0
    rounded_count_mismatch = 0
    max_pct_reconstruction_error = 0.0
    probes = set()

    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if line_no == 1 and is_header(parts):
                continue

            if len(parts) < 6:
                raise ValueError(
                    f"{path}: expected >=6 columns at line {line_no}, got {len(parts)}"
                )

            chrom, start, end, cov_s, pct_s, probe_id = parts[:6]

            try:
                cov = int(float(cov_s))
                pct = float(pct_s)
            except ValueError as exc:
                raise ValueError(f"{path}: invalid numeric value at line {line_no}") from exc

            if cov < 0:
                raise ValueError(f"{path}: negative coverage at line {line_no}")
            if not (0.0 <= pct <= 100.0):
                raise ValueError(f"{path}: methylation percentage outside [0,100] at line {line_no}")

            n_loci += 1
            total_cov += cov
            probes.add(probe_id)

            cov_gt1 += cov > 1
            cov_ge2 += cov >= 2
            cov_ge5 += cov >= 5
            cov_ge10 += cov >= 10
            cov_ge20 += cov >= 20

            # Percentage is rounded to two decimals in the public files.
            # Recover the nearest integer number of methylated reads.
            meth = int(round(cov * pct / 100.0)) if cov > 0 else 0
            meth = max(0, min(cov, meth))
            unmeth = cov - meth
            methylated_sum += meth
            unmethylated_sum += unmeth

            if cov > 0:
                reconstructed_pct = 100.0 * meth / cov
                err = abs(reconstructed_pct - pct)
                max_pct_reconstruction_error = max(max_pct_reconstruction_error, err)

                # With percentages rounded to two decimals, an error >0.011 is suspicious.
                if err > 0.011:
                    rounded_count_mismatch += 1

    mean_cov = total_cov / n_loci if n_loci else float("nan")

    return {
        "n_loci": n_loci,
        "n_unique_probes": len(probes),
        "total_coverage": total_cov,
        "mean_coverage_per_locus": mean_cov,
        "frac_cov_gt1": cov_gt1 / n_loci if n_loci else float("nan"),
        "frac_cov_ge2": cov_ge2 / n_loci if n_loci else float("nan"),
        "frac_cov_ge5": cov_ge5 / n_loci if n_loci else float("nan"),
        "frac_cov_ge10": cov_ge10 / n_loci if n_loci else float("nan"),
        "frac_cov_ge20": cov_ge20 / n_loci if n_loci else float("nan"),
        "estimated_methylated_reads": methylated_sum,
        "estimated_unmethylated_reads": unmethylated_sum,
        "integer_reconstruction_mismatches": rounded_count_mismatch,
        "max_pct_reconstruction_error": max_pct_reconstruction_error,
        "probes": probes,
    }


def fmt(x):
    if isinstance(x, float):
        if math.isnan(x):
            return "NA"
        return f"{x:.8g}"
    return str(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", type=Path)
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    audit_dir = data_root / "registry" / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    qc_rows = []
    cohort_probe_sets = {}

    for cohort, rel in COHORTS.items():
        cohort_dir = data_root / rel
        if not cohort_dir.exists():
            raise FileNotFoundError(f"Missing cohort directory: {cohort_dir}")

        files = sorted(
            p for p in cohort_dir.iterdir()
            if p.is_file() and not p.name.startswith(".")
        )
        if not files:
            raise RuntimeError(f"No files found in {cohort_dir}")

        union = set()
        intersection = None

        print(f"[{cohort}] auditing {len(files)} files")
        for idx, path in enumerate(files, start=1):
            stats = parse_file(path)
            probes = stats.pop("probes")

            union.update(probes)
            intersection = probes.copy() if intersection is None else intersection.intersection(probes)

            row = {
                "cohort": cohort,
                "sample_file": path.name,
                **stats,
            }
            qc_rows.append(row)

            if idx == 1 or idx % 25 == 0 or idx == len(files):
                print(
                    f"  {idx:>3}/{len(files)}  {path.name}  "
                    f"loci={stats['n_loci']:,} mean_cov={stats['mean_coverage_per_locus']:.3f}"
                )

        cohort_probe_sets[cohort] = {
            "union": union,
            "intersection": intersection if intersection is not None else set(),
            "n_files": len(files),
        }

    qc_path = audit_dir / "rapidcns2_methylation_qc.tsv"
    fieldnames = list(qc_rows[0].keys())
    with qc_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in qc_rows:
            writer.writerow({k: fmt(v) for k, v in row.items()})

    probe_space_path = audit_dir / "rapidcns2_probe_space.tsv"
    with probe_space_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(
            [
                "cohort",
                "n_files",
                "n_probe_union",
                "n_probe_intersection_all_samples",
                "intersection_over_union",
            ]
        )
        for cohort, d in cohort_probe_sets.items():
            union = d["union"]
            inter = d["intersection"]
            writer.writerow(
                [
                    cohort,
                    d["n_files"],
                    len(union),
                    len(inter),
                    fmt(len(inter) / len(union) if union else float("nan")),
                ]
            )

    a = cohort_probe_sets["ONT_WGS"]
    b = cohort_probe_sets["Rapid_CNS2"]

    overlap_path = audit_dir / "rapidcns2_probe_overlap.tsv"
    with overlap_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["metric", "value"])
        writer.writerow(["ONT_WGS_union", len(a["union"])])
        writer.writerow(["Rapid_CNS2_union", len(b["union"])])
        writer.writerow(["union_overlap", len(a["union"] & b["union"])])
        writer.writerow(["union_combined", len(a["union"] | b["union"])])
        writer.writerow(
            [
                "union_jaccard",
                fmt(
                    len(a["union"] & b["union"]) / len(a["union"] | b["union"])
                    if (a["union"] | b["union"])
                    else float("nan")
                ),
            ]
        )
        writer.writerow(
            [
                "shared_by_all_samples_both_cohorts",
                len(a["intersection"] & b["intersection"]),
            ]
        )

    print()
    print("Wrote:")
    print(f"  {qc_path}")
    print(f"  {probe_space_path}")
    print(f"  {overlap_path}")


if __name__ == "__main__":
    main()
