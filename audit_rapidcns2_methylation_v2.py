#!/usr/bin/env python3
"""
Audit Rapid-CNS2 / ONT-WGS methylation files.

Read-only audit. Missing methylation values (NA/NaN/.) are tolerated and
explicitly counted rather than causing the audit to fail.

Usage:
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

MISSING = {"na", "nan", ".", "", "null", "none"}


def is_missing(x):
    return x.strip().lower() in MISSING


def is_header(parts):
    if not parts:
        return False
    low = [p.lower() for p in parts]
    return parts[0].lower() in {"chr", "chrom", "chromosome"} and "coverage" in low


def parse_file(path):
    n_rows = 0
    n_valid_methylation = 0
    n_missing_methylation = 0
    n_missing_coverage = 0

    total_cov_all = 0
    total_cov_valid_meth = 0

    cov_gt1 = cov_ge2 = cov_ge5 = cov_ge10 = cov_ge20 = 0

    methylated_sum = 0
    unmethylated_sum = 0
    rounded_count_mismatch = 0
    max_pct_reconstruction_error = 0.0

    probes_all = set()
    probes_valid_meth = set()

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
            n_rows += 1
            probes_all.add(probe_id)

            # Coverage may theoretically also be missing; tolerate and count it.
            if is_missing(cov_s):
                n_missing_coverage += 1
                cov = None
            else:
                try:
                    cov = int(float(cov_s))
                except ValueError as exc:
                    raise ValueError(
                        f"{path}: invalid coverage '{cov_s}' at line {line_no}"
                    ) from exc

                if cov < 0:
                    raise ValueError(f"{path}: negative coverage at line {line_no}")

                total_cov_all += cov
                cov_gt1 += cov > 1
                cov_ge2 += cov >= 2
                cov_ge5 += cov >= 5
                cov_ge10 += cov >= 10
                cov_ge20 += cov >= 20

            # Missing methylation percentage is a valid QC state, not a parser error.
            if is_missing(pct_s):
                n_missing_methylation += 1
                continue

            try:
                pct = float(pct_s)
            except ValueError as exc:
                raise ValueError(
                    f"{path}: invalid methylation percentage '{pct_s}' at line {line_no}"
                ) from exc

            if not (0.0 <= pct <= 100.0):
                raise ValueError(
                    f"{path}: methylation percentage outside [0,100] at line {line_no}"
                )

            n_valid_methylation += 1
            probes_valid_meth.add(probe_id)

            if cov is None:
                # Percentage can still be audited as present, but read counts cannot be reconstructed.
                continue

            total_cov_valid_meth += cov

            # Public percentages are rounded to two decimals. Recover nearest integer count.
            meth = int(round(cov * pct / 100.0)) if cov > 0 else 0
            meth = max(0, min(cov, meth))
            unmeth = cov - meth

            methylated_sum += meth
            unmethylated_sum += unmeth

            if cov > 0:
                reconstructed_pct = 100.0 * meth / cov
                err = abs(reconstructed_pct - pct)
                max_pct_reconstruction_error = max(max_pct_reconstruction_error, err)

                # Two-decimal rounding should ordinarily produce <= 0.005 error.
                # Use a slightly looser threshold for floating-point safety.
                if err > 0.011:
                    rounded_count_mismatch += 1

    n_cov_valid = n_rows - n_missing_coverage

    return {
        "n_rows": n_rows,
        "n_unique_probes_all": len(probes_all),
        "n_valid_methylation": n_valid_methylation,
        "n_missing_methylation": n_missing_methylation,
        "frac_missing_methylation": (
            n_missing_methylation / n_rows if n_rows else float("nan")
        ),
        "n_missing_coverage": n_missing_coverage,
        "total_coverage_all_rows": total_cov_all,
        "mean_coverage_all_nonmissing": (
            total_cov_all / n_cov_valid if n_cov_valid else float("nan")
        ),
        "mean_coverage_valid_methylation": (
            total_cov_valid_meth / n_valid_methylation
            if n_valid_methylation else float("nan")
        ),
        "frac_cov_gt1": cov_gt1 / n_cov_valid if n_cov_valid else float("nan"),
        "frac_cov_ge2": cov_ge2 / n_cov_valid if n_cov_valid else float("nan"),
        "frac_cov_ge5": cov_ge5 / n_cov_valid if n_cov_valid else float("nan"),
        "frac_cov_ge10": cov_ge10 / n_cov_valid if n_cov_valid else float("nan"),
        "frac_cov_ge20": cov_ge20 / n_cov_valid if n_cov_valid else float("nan"),
        "estimated_methylated_reads": methylated_sum,
        "estimated_unmethylated_reads": unmethylated_sum,
        "integer_reconstruction_mismatches": rounded_count_mismatch,
        "max_pct_reconstruction_error": max_pct_reconstruction_error,
        "probes_all": probes_all,
        "probes_valid_meth": probes_valid_meth,
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
            if p.is_file()
            and not p.name.startswith(".")
            and p.suffix.lower() == ".txt"
        )
        if not files:
            raise RuntimeError(f"No .txt files found in {cohort_dir}")

        union_all = set()
        intersection_all = None
        union_valid = set()
        intersection_valid = None

        print(f"[{cohort}] auditing {len(files)} files")

        for idx, path in enumerate(files, start=1):
            stats = parse_file(path)
            probes_all = stats.pop("probes_all")
            probes_valid = stats.pop("probes_valid_meth")

            union_all.update(probes_all)
            intersection_all = (
                probes_all.copy()
                if intersection_all is None
                else intersection_all.intersection(probes_all)
            )

            union_valid.update(probes_valid)
            intersection_valid = (
                probes_valid.copy()
                if intersection_valid is None
                else intersection_valid.intersection(probes_valid)
            )

            qc_rows.append({
                "cohort": cohort,
                "sample_file": path.name,
                **stats,
            })

            if idx == 1 or idx % 25 == 0 or idx == len(files):
                print(
                    f"  {idx:>3}/{len(files)}  {path.name}  "
                    f"rows={stats['n_rows']:,} "
                    f"missing_meth={stats['n_missing_methylation']:,} "
                    f"mean_cov={stats['mean_coverage_all_nonmissing']:.3f}"
                )

        cohort_probe_sets[cohort] = {
            "union_all": union_all,
            "intersection_all": intersection_all or set(),
            "union_valid": union_valid,
            "intersection_valid": intersection_valid or set(),
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
        writer.writerow([
            "cohort",
            "n_files",
            "n_probe_union_all_rows",
            "n_probe_intersection_all_rows",
            "n_probe_union_valid_methylation",
            "n_probe_intersection_valid_methylation",
        ])
        for cohort, d in cohort_probe_sets.items():
            writer.writerow([
                cohort,
                d["n_files"],
                len(d["union_all"]),
                len(d["intersection_all"]),
                len(d["union_valid"]),
                len(d["intersection_valid"]),
            ])

    a = cohort_probe_sets["ONT_WGS"]
    b = cohort_probe_sets["Rapid_CNS2"]

    overlap_path = audit_dir / "rapidcns2_probe_overlap.tsv"
    with overlap_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["metric", "value"])

        pairs = [
            ("ONT_WGS_union_valid", len(a["union_valid"])),
            ("Rapid_CNS2_union_valid", len(b["union_valid"])),
            ("valid_union_overlap", len(a["union_valid"] & b["union_valid"])),
            ("valid_union_combined", len(a["union_valid"] | b["union_valid"])),
            (
                "valid_union_jaccard",
                len(a["union_valid"] & b["union_valid"]) / len(a["union_valid"] | b["union_valid"])
                if (a["union_valid"] | b["union_valid"]) else float("nan")
            ),
            (
                "shared_valid_in_all_samples_both_cohorts",
                len(a["intersection_valid"] & b["intersection_valid"]),
            ),
        ]
        for key, value in pairs:
            writer.writerow([key, fmt(value)])

    print()
    print("Wrote:")
    print(f"  {qc_path}")
    print(f"  {probe_space_path}")
    print(f"  {overlap_path}")


if __name__ == "__main__":
    main()
