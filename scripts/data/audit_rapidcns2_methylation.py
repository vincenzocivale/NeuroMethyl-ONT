#!/usr/bin/env python3
"""
Schema-aware audit for Rapid-CNS2 / ONT-WGS methylation files.

Design goals
------------
- Read-only: never modifies source files.
- Detects headers even when they are not on line 1.
- Supports:
    1) canonical MNP-Flex schema:
       chr start end coverage methylation_percentage IlmnID
    2) count schema:
       chr start end m u IlmnID
       -> coverage = m + u
       -> methylation_percentage = 100 * m / (m + u)
- Tolerates explicit missing values (NA/NaN/./null).
- Does NOT silently swallow malformed data rows:
  anomalies are written to a TSV with file + line number + reason.
- Records the detected schema for every sample.
- Reports missing numeric sample IDs in ONT_WGS / Rapid_CNS2 file series.

Usage
-----
python scripts/data/audit_rapidcns2_methylation.py /path/to/CNSNanoporeData
"""

import argparse
import csv
import math
import re
from pathlib import Path

COHORTS = {
    "ONT_WGS": {
        "relpath": "datasets/RapidCNS2/raw/ONT_WGS/extracted/ONT_WGS",
        "id_regex": re.compile(r"ONT_WGS_(\d+)\.txt$", re.I),
    },
    "Rapid_CNS2": {
        "relpath": "datasets/RapidCNS2/raw/Rapid_CNS2/extracted/Rapid-CNS2",
        "id_regex": re.compile(r"ID_(\d+)\.bed\.txt$", re.I),
    },
}

MISSING = {"na", "nan", ".", "", "null", "none"}

ALIASES = {
    "chrom": {"chr", "chrom", "chromosome"},
    "start": {"start", "chromstart", "chrom_start"},
    "end": {"end", "chromend", "chrom_end"},
    "coverage": {"coverage", "cov", "depth", "n"},
    "pct": {
        "methylation_percentage",
        "methylation_percent",
        "methylationpercentage",
        "percent_methylated",
        "pct_methylated",
        "beta",
    },
    "probe": {"ilmnid", "ilmn_id", "probe_id", "probe", "cpg", "cpg_id"},
    "m": {"m", "meth", "methylated", "methylated_reads", "modified"},
    "u": {"u", "unmeth", "unmethylated", "unmethylated_reads", "canonical"},
}


def norm_token(x):
    return x.strip().lower().replace("%", "percent").replace("-", "_")


def is_missing(x):
    return norm_token(x) in MISSING


def is_number(x):
    if is_missing(x):
        return False
    try:
        float(x)
        return True
    except ValueError:
        return False


def map_header(parts):
    """Return (schema_name, indices_dict) if line is a recognized header, else None."""
    toks = [norm_token(x) for x in parts]

    def find(alias_key):
        aliases = ALIASES[alias_key]
        for i, t in enumerate(toks):
            if t in aliases:
                return i
        return None

    chrom = find("chrom")
    start = find("start")
    end = find("end")
    probe = find("probe")
    cov = find("coverage")
    pct = find("pct")
    m = find("m")
    u = find("u")

    # Canonical MNP-Flex representation.
    if cov is not None and pct is not None and probe is not None:
        return (
            "coverage_percentage",
            {
                "chrom": chrom,
                "start": start,
                "end": end,
                "coverage": cov,
                "pct": pct,
                "probe": probe,
            },
        )

    # Explicit methylated/unmethylated counts.
    if m is not None and u is not None and probe is not None:
        return (
            "m_u_counts",
            {
                "chrom": chrom,
                "start": start,
                "end": end,
                "m": m,
                "u": u,
                "probe": probe,
            },
        )

    return None


def default_schema_for_six_columns():
    return (
        "coverage_percentage_headerless",
        {
            "chrom": 0,
            "start": 1,
            "end": 2,
            "coverage": 3,
            "pct": 4,
            "probe": 5,
        },
    )


def safe_get(parts, idx):
    if idx is None:
        return ""
    if idx >= len(parts):
        raise IndexError(f"column index {idx} >= row width {len(parts)}")
    return parts[idx]


def parse_standard_row(parts, idx):
    cov_s = safe_get(parts, idx["coverage"])
    pct_s = safe_get(parts, idx["pct"])
    probe = safe_get(parts, idx["probe"])

    cov = None if is_missing(cov_s) else int(float(cov_s))
    pct = None if is_missing(pct_s) else float(pct_s)

    if cov is not None and cov < 0:
        raise ValueError("negative coverage")
    if pct is not None and not (0 <= pct <= 100):
        # beta can occasionally be stored 0..1; do not auto-convert silently.
        raise ValueError(f"methylation percentage outside [0,100]: {pct}")

    return cov, pct, probe, None, None


def parse_mu_row(parts, idx):
    m_s = safe_get(parts, idx["m"])
    u_s = safe_get(parts, idx["u"])
    probe = safe_get(parts, idx["probe"])

    m = None if is_missing(m_s) else int(float(m_s))
    u = None if is_missing(u_s) else int(float(u_s))

    if m is not None and m < 0:
        raise ValueError("negative methylated-read count")
    if u is not None and u < 0:
        raise ValueError("negative unmethylated-read count")

    if m is None or u is None:
        return None, None, probe, m, u

    cov = m + u
    pct = (100.0 * m / cov) if cov > 0 else None
    return cov, pct, probe, m, u


def parse_file(path, anomalies):
    schema_name = None
    idx = None
    detected_header_line = None

    n_rows = 0
    n_valid_methylation = 0
    n_missing_methylation = 0
    n_missing_coverage = 0
    n_malformed_rows = 0
    n_header_rows = 0

    total_cov = 0
    total_cov_valid_meth = 0

    cov_gt1 = cov_ge2 = cov_ge5 = cov_ge10 = cov_ge20 = 0

    methylated_sum = 0
    unmethylated_sum = 0
    integer_reconstruction_mismatches = 0
    max_pct_reconstruction_error = 0.0

    probes_all = set()
    probes_valid = set()

    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line_no, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue

            parts = line.split()

            # Header detection is allowed anywhere in the file.
            hdr = map_header(parts)
            if hdr is not None:
                new_schema, new_idx = hdr
                n_header_rows += 1

                if schema_name is None:
                    schema_name = new_schema
                    idx = new_idx
                    detected_header_line = line_no
                elif schema_name != new_schema:
                    anomalies.append(
                        (
                            path.name,
                            line_no,
                            "schema_change",
                            f"{schema_name} -> {new_schema}",
                            line[:500],
                        )
                    )
                    schema_name = new_schema
                    idx = new_idx
                # Header is metadata, never a biological row.
                continue

            # If no header has been seen, only assume the documented six-column layout
            # when the first putative data row looks numeric in start/end/coverage.
            if schema_name is None:
                if (
                    len(parts) >= 6
                    and is_number(parts[1])
                    and is_number(parts[2])
                    and (is_number(parts[3]) or is_missing(parts[3]))
                ):
                    schema_name, idx = default_schema_for_six_columns()
                else:
                    anomalies.append(
                        (
                            path.name,
                            line_no,
                            "unrecognized_preamble_or_header",
                            "no schema detected yet",
                            line[:500],
                        )
                    )
                    continue

            try:
                if schema_name.startswith("coverage_percentage"):
                    cov, pct, probe, m_explicit, u_explicit = parse_standard_row(parts, idx)
                elif schema_name == "m_u_counts":
                    cov, pct, probe, m_explicit, u_explicit = parse_mu_row(parts, idx)
                else:
                    raise RuntimeError(f"unsupported internal schema: {schema_name}")
            except Exception as exc:
                n_malformed_rows += 1
                anomalies.append(
                    (
                        path.name,
                        line_no,
                        "malformed_data_row",
                        str(exc),
                        line[:500],
                    )
                )
                continue

            n_rows += 1
            probes_all.add(probe)

            if cov is None:
                n_missing_coverage += 1
            else:
                total_cov += cov
                cov_gt1 += cov > 1
                cov_ge2 += cov >= 2
                cov_ge5 += cov >= 5
                cov_ge10 += cov >= 10
                cov_ge20 += cov >= 20

            if pct is None:
                n_missing_methylation += 1
                continue

            n_valid_methylation += 1
            probes_valid.add(probe)

            if cov is None:
                continue

            total_cov_valid_meth += cov

            # If explicit m/u counts exist, use them exactly.
            if m_explicit is not None and u_explicit is not None:
                meth = m_explicit
                unmeth = u_explicit
            else:
                # Otherwise reconstruct nearest integer read counts from rounded percent.
                meth = int(round(cov * pct / 100.0)) if cov > 0 else 0
                meth = max(0, min(cov, meth))
                unmeth = cov - meth

                if cov > 0:
                    reconstructed_pct = 100.0 * meth / cov
                    err = abs(reconstructed_pct - pct)
                    max_pct_reconstruction_error = max(
                        max_pct_reconstruction_error, err
                    )
                    if err > 0.011:
                        integer_reconstruction_mismatches += 1

            methylated_sum += meth
            unmethylated_sum += unmeth

    if schema_name is None:
        schema_name = "undetected"

    n_cov_valid = n_rows - n_missing_coverage

    return {
        "detected_schema": schema_name,
        "detected_header_line": detected_header_line or "",
        "n_header_rows": n_header_rows,
        "n_rows": n_rows,
        "n_malformed_rows": n_malformed_rows,
        "n_unique_probes_all": len(probes_all),
        "n_valid_methylation": n_valid_methylation,
        "n_missing_methylation": n_missing_methylation,
        "frac_missing_methylation": (
            n_missing_methylation / n_rows if n_rows else float("nan")
        ),
        "n_missing_coverage": n_missing_coverage,
        "total_coverage_all_rows": total_cov,
        "mean_coverage_all_nonmissing": (
            total_cov / n_cov_valid if n_cov_valid else float("nan")
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
        "integer_reconstruction_mismatches": integer_reconstruction_mismatches,
        "max_pct_reconstruction_error": max_pct_reconstruction_error,
        "probes_all": probes_all,
        "probes_valid": probes_valid,
    }


def fmt(x):
    if isinstance(x, float):
        if math.isnan(x):
            return "NA"
        return f"{x:.8g}"
    return str(x)


def extract_numeric_id(path, regex):
    m = regex.search(path.name)
    return int(m.group(1)) if m else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", type=Path)
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    audit_dir = data_root / "registry" / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    qc_rows = []
    anomalies = []
    cohort_probe_sets = {}
    manifest_rows = []

    for cohort, cfg in COHORTS.items():
        cohort_dir = data_root / cfg["relpath"]
        if not cohort_dir.exists():
            raise FileNotFoundError(f"Missing cohort directory: {cohort_dir}")

        files = sorted(
            [
                p
                for p in cohort_dir.iterdir()
                if p.is_file()
                and not p.name.startswith(".")
                and p.suffix.lower() == ".txt"
            ],
            key=lambda p: (
                extract_numeric_id(p, cfg["id_regex"]) is None,
                extract_numeric_id(p, cfg["id_regex"]) or 10**12,
                p.name,
            ),
        )

        if not files:
            raise RuntimeError(f"No .txt files found in {cohort_dir}")

        ids = [
            extract_numeric_id(p, cfg["id_regex"])
            for p in files
            if extract_numeric_id(p, cfg["id_regex"]) is not None
        ]
        missing_ids = []
        if ids:
            missing_ids = sorted(set(range(min(ids), max(ids) + 1)) - set(ids))

        union_all = set()
        intersection_all = None
        union_valid = set()
        intersection_valid = None

        print(f"[{cohort}] auditing {len(files)} files")
        if missing_ids:
            print(f"  numeric file-ID gaps: {missing_ids}")

        for idx_file, path in enumerate(files, start=1):
            stats = parse_file(path, anomalies)
            probes_all = stats.pop("probes_all")
            probes_valid = stats.pop("probes_valid")

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

            numeric_id = extract_numeric_id(path, cfg["id_regex"])
            qc_rows.append(
                {
                    "cohort": cohort,
                    "sample_file": path.name,
                    "numeric_file_id": numeric_id if numeric_id is not None else "",
                    **stats,
                }
            )

            manifest_rows.append(
                {
                    "cohort": cohort,
                    "sample_file": path.name,
                    "numeric_file_id": numeric_id if numeric_id is not None else "",
                    "size_bytes": path.stat().st_size,
                    "detected_schema": stats["detected_schema"],
                }
            )

            if idx_file == 1 or idx_file % 25 == 0 or idx_file == len(files):
                print(
                    f"  {idx_file:>3}/{len(files)}  {path.name}  "
                    f"schema={stats['detected_schema']} "
                    f"rows={stats['n_rows']:,} "
                    f"malformed={stats['n_malformed_rows']:,} "
                    f"missing_meth={stats['n_missing_methylation']:,} "
                    f"mean_cov={stats['mean_coverage_all_nonmissing']:.3f}"
                )

        cohort_probe_sets[cohort] = {
            "union_all": union_all,
            "intersection_all": intersection_all or set(),
            "union_valid": union_valid,
            "intersection_valid": intersection_valid or set(),
            "n_files": len(files),
            "missing_ids": missing_ids,
        }

    qc_path = audit_dir / "rapidcns2_methylation_qc.tsv"
    with qc_path.open("w", encoding="utf-8", newline="") as fh:
        fields = list(qc_rows[0].keys())
        writer = csv.DictWriter(fh, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in qc_rows:
            writer.writerow({k: fmt(v) for k, v in row.items()})

    anomalies_path = audit_dir / "rapidcns2_parse_anomalies.tsv"
    with anomalies_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["sample_file", "line_number", "kind", "reason", "line_preview"])
        writer.writerows(anomalies)

    manifest_path = audit_dir / "rapidcns2_sample_file_manifest.tsv"
    with manifest_path.open("w", encoding="utf-8", newline="") as fh:
        fields = list(manifest_rows[0].keys())
        writer = csv.DictWriter(fh, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(manifest_rows)

    probe_space_path = audit_dir / "rapidcns2_probe_space.tsv"
    with probe_space_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(
            [
                "cohort",
                "n_files",
                "missing_numeric_file_ids",
                "n_probe_union_all_rows",
                "n_probe_intersection_all_rows",
                "n_probe_union_valid_methylation",
                "n_probe_intersection_valid_methylation",
            ]
        )
        for cohort, d in cohort_probe_sets.items():
            writer.writerow(
                [
                    cohort,
                    d["n_files"],
                    ",".join(map(str, d["missing_ids"])),
                    len(d["union_all"]),
                    len(d["intersection_all"]),
                    len(d["union_valid"]),
                    len(d["intersection_valid"]),
                ]
            )

    a = cohort_probe_sets["ONT_WGS"]
    b = cohort_probe_sets["Rapid_CNS2"]

    overlap_path = audit_dir / "rapidcns2_probe_overlap.tsv"
    with overlap_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["metric", "value"])
        combined = a["union_valid"] | b["union_valid"]
        overlap = a["union_valid"] & b["union_valid"]
        metrics = [
            ("ONT_WGS_union_valid", len(a["union_valid"])),
            ("Rapid_CNS2_union_valid", len(b["union_valid"])),
            ("valid_union_overlap", len(overlap)),
            ("valid_union_combined", len(combined)),
            (
                "valid_union_jaccard",
                len(overlap) / len(combined) if combined else float("nan"),
            ),
            (
                "shared_valid_in_all_samples_both_cohorts",
                len(a["intersection_valid"] & b["intersection_valid"]),
            ),
        ]
        for key, value in metrics:
            writer.writerow([key, fmt(value)])

    print()
    print("Wrote:")
    print(f"  {qc_path}")
    print(f"  {anomalies_path}")
    print(f"  {manifest_path}")
    print(f"  {probe_space_path}")
    print(f"  {overlap_path}")
    print(f"  anomalies recorded: {len(anomalies):,}")


if __name__ == "__main__":
    main()
