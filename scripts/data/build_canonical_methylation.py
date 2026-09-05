#!/usr/bin/env python3
"""Build canonical coordinate-level and probe-level methylation Parquet outputs
from raw Rapid-CNS2 / ONT-WGS methylation files.

    raw sample file
        -> read_rapidcns2_sample()            (schema-aware parse)
        -> to_coordinate_level()               (+ sample_id, canonical columns)
        -> validate_* invariants
        -> aggregate_to_probe_level()          (evidence-weighted, sum m/u)
        -> validate_probe_aggregation()
        -> write coordinate + probe Parquet
        -> update manifest row

Never modifies raw files or existing audit outputs. By default, existing
derived outputs are not overwritten (pass --overwrite to force). A schema
that cannot be identified, or an invariant violation, fails that sample
loudly (logged as ERROR, recorded in the manifest with status=failed) but
does not abort the run.

Usage:
    python scripts/data/build_canonical_methylation.py "$DATA_ROOT" \\
        --dataset RapidCNS2 [--cohort ONT_WGS|Rapid_CNS2] [--sample ID_1] \\
        [--overwrite] [--dry-run]
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from neuromethyl_ont.data.canonical import (  # noqa: E402
    aggregate_to_probe_level,
    to_coordinate_level,
)
from neuromethyl_ont.data.rapidcns2 import (  # noqa: E402
    RapidCNS2ParseError,
    SchemaDetectionError,
    read_rapidcns2_sample,
)
from neuromethyl_ont.data.validation import (  # noqa: E402
    validate_beta_range,
    validate_coordinates,
    validate_counts,
    validate_probe_aggregation,
)

try:
    PIPELINE_VERSION = version("neuromethyl-ont")
except PackageNotFoundError:
    PIPELINE_VERSION = "0.0.0+unknown"

COHORTS = {
    "ONT_WGS": {
        "relpath": Path("datasets/RapidCNS2/raw/ONT_WGS/extracted/ONT_WGS"),
        "id_regex": re.compile(r"(ONT_WGS_\d+)\.txt$", re.I),
        "glob": "*.txt",
    },
    "Rapid_CNS2": {
        "relpath": Path("datasets/RapidCNS2/raw/Rapid_CNS2/extracted/Rapid-CNS2"),
        "id_regex": re.compile(r"(ID_\d+)\.bed\.txt$", re.I),
        "glob": "*.bed.txt",
    },
}

MANIFEST_FIELDS = [
    "dataset_id",
    "cohort",
    "sample_id",
    "source_file",
    "coordinate_output",
    "probe_output",
    "n_source_rows",
    "n_coordinate_rows",
    "n_probe_rows",
    "total_coverage_source",
    "total_coverage_probe",
    "n_missing",
    "parser_schema",
    "created_at",
    "pipeline_version",
    "status",
    "error_message",
]

log = logging.getLogger("build_canonical_methylation")


def setup_logging(repo_root: Path) -> Path:
    log_dir = repo_root / "outputs" / "logs" / "data"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = log_dir / f"build_canonical_methylation_{stamp}.log"

    log.setLevel(logging.INFO)
    log.handlers.clear()

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    log.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    log.addHandler(ch)

    return log_path


def extract_sample_id(path: Path, id_regex: re.Pattern) -> str:
    m = id_regex.search(path.name)
    if not m:
        raise ValueError(f"cannot extract sample_id from filename: {path.name}")
    return m.group(1)


def list_cohort_files(data_root: Path, cohort: str, sample_filter: str | None) -> list[Path]:
    cfg = COHORTS[cohort]
    cohort_dir = data_root / cfg["relpath"]
    if not cohort_dir.exists():
        raise FileNotFoundError(f"Missing cohort directory: {cohort_dir}")

    files = sorted(
        p for p in cohort_dir.glob(cfg["glob"])
        if p.is_file() and not p.name.startswith(".")
    )
    if sample_filter is not None:
        files = [
            p for p in files
            if extract_sample_id(p, cfg["id_regex"]) == sample_filter
        ]
    return files


def load_existing_manifest(manifest_path: Path) -> dict[tuple[str, str, str], dict]:
    if not manifest_path.exists():
        return {}
    rows: dict[tuple[str, str, str], dict] = {}
    with manifest_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            key = (row["dataset_id"], row["cohort"], row["sample_id"])
            rows[key] = row
    return rows


def write_manifest(manifest_path: Path, rows_by_key: dict[tuple[str, str, str], dict]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows_by_key.values(), key=lambda r: (r["dataset_id"], r["cohort"], r["sample_id"]))
    with manifest_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS, delimiter="\t")
        writer.writeheader()
        for row in ordered:
            writer.writerow({k: row.get(k, "") for k in MANIFEST_FIELDS})


def process_sample(
    source_file: Path,
    cohort: str,
    sample_id: str,
    coord_out: Path,
    probe_out: Path,
    dry_run: bool,
) -> dict:
    manifest_row = {
        "dataset_id": "RapidCNS2",
        "cohort": cohort,
        "sample_id": sample_id,
        "source_file": str(source_file),
        "coordinate_output": str(coord_out),
        "probe_output": str(probe_out),
        "n_source_rows": "",
        "n_coordinate_rows": "",
        "n_probe_rows": "",
        "total_coverage_source": "",
        "total_coverage_probe": "",
        "n_missing": "",
        "parser_schema": "",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": PIPELINE_VERSION,
        "status": "failed",
        "error_message": "",
    }

    try:
        raw_df, stats = read_rapidcns2_sample(source_file)
        manifest_row["parser_schema"] = stats.detected_schema
        manifest_row["n_source_rows"] = stats.n_data_rows + stats.n_malformed_rows
        manifest_row["n_missing"] = stats.n_missing_methylation

        if stats.n_malformed_rows:
            log.warning(
                "%s/%s: %d malformed row(s) excluded (see anomalies)",
                cohort, sample_id, stats.n_malformed_rows,
            )
        if stats.special_probe_counts:
            log.info(
                "%s/%s: non-cg probe annotations: %s",
                cohort, sample_id,
                ", ".join(f"{k}={v}" for k, v in sorted(stats.special_probe_counts.items())),
            )

        coord_df = to_coordinate_level(raw_df, sample_id=sample_id)

        violations = (
            validate_coordinates(coord_df)
            + validate_counts(coord_df)
            + validate_beta_range(coord_df)
        )
        if violations:
            manifest_row["error_message"] = "coordinate-level validation failed: " + "; ".join(violations)
            log.error("%s/%s: %s", cohort, sample_id, manifest_row["error_message"])
            return manifest_row

        probe_df = aggregate_to_probe_level(coord_df)
        probe_violations = validate_probe_aggregation(coord_df, probe_df)
        if probe_violations:
            manifest_row["error_message"] = "probe-level validation failed: " + "; ".join(probe_violations)
            log.error("%s/%s: %s", cohort, sample_id, manifest_row["error_message"])
            return manifest_row

        manifest_row["n_coordinate_rows"] = len(coord_df)
        manifest_row["n_probe_rows"] = len(probe_df)
        # "Total coverage" is defined over rows that carry actual m/u evidence,
        # matching validate_probe_aggregation -- coverage recorded on a row
        # whose methylation percentage is missing contributes no evidence and
        # is intentionally excluded from both sides of this comparison.
        has_evidence = coord_df["methylated_count"].notna() & coord_df["unmethylated_count"].notna()
        manifest_row["total_coverage_source"] = int(coord_df.loc[has_evidence, "coverage"].sum(skipna=True) or 0)
        manifest_row["total_coverage_probe"] = int(probe_df["coverage"].sum(skipna=True) or 0)

        if not dry_run:
            coord_out.parent.mkdir(parents=True, exist_ok=True)
            probe_out.parent.mkdir(parents=True, exist_ok=True)
            coord_df.to_parquet(coord_out, index=False)
            probe_df.to_parquet(probe_out, index=False)

        manifest_row["status"] = "dry_run_ok" if dry_run else "ok"
        log.info(
            "%s/%s: OK schema=%s coordinate_rows=%d probe_rows=%d "
            "coverage_preserved=%s total_coverage=%d",
            cohort, sample_id, stats.detected_schema,
            len(coord_df), len(probe_df),
            manifest_row["total_coverage_source"] == manifest_row["total_coverage_probe"],
            manifest_row["total_coverage_source"],
        )
        return manifest_row

    except (SchemaDetectionError, RapidCNS2ParseError) as exc:
        manifest_row["error_message"] = str(exc)
        log.error("%s/%s: FAILED (schema): %s", cohort, sample_id, exc)
        return manifest_row
    except Exception as exc:  # noqa: BLE001 - a single failed sample must not abort the run
        manifest_row["error_message"] = f"unexpected error: {exc}"
        log.exception("%s/%s: FAILED (unexpected)", cohort, sample_id)
        return manifest_row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_root", type=Path)
    ap.add_argument("--dataset", choices=["RapidCNS2"], default="RapidCNS2")
    ap.add_argument("--cohort", choices=list(COHORTS), action="append", default=None)
    ap.add_argument("--sample", default=None, help="Process only this sample_id (e.g. ID_1, ONT_WGS_10)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing derived Parquet outputs")
    ap.add_argument("--dry-run", action="store_true", help="Parse and validate but write nothing")
    args = ap.parse_args()

    log_path = setup_logging(REPO_ROOT)
    log.info("Log file: %s", log_path)

    data_root = args.data_root.resolve()
    cohorts = args.cohort or list(COHORTS)

    manifest_path = data_root / "registry" / "derived" / "canonical_methylation_manifest.tsv"
    manifest_rows = load_existing_manifest(manifest_path)

    n_ok = n_failed = n_skipped = 0

    for cohort in cohorts:
        try:
            files = list_cohort_files(data_root, cohort, args.sample)
        except FileNotFoundError as exc:
            log.error(str(exc))
            return 1

        if not files:
            log.warning("cohort=%s: no files matched (sample filter=%s)", cohort, args.sample)
            continue

        log.info("cohort=%s: %d file(s) to consider", cohort, len(files))

        for i, source_file in enumerate(files, start=1):
            sample_id = extract_sample_id(source_file, COHORTS[cohort]["id_regex"])
            coord_out = (
                data_root / "derived" / "ont_native" / "coordinate_level"
                / "RapidCNS2" / cohort / f"{sample_id}.parquet"
            )
            probe_out = (
                data_root / "derived" / "array_aligned" / "probe_level"
                / "RapidCNS2" / cohort / f"{sample_id}.parquet"
            )

            if not args.overwrite and coord_out.exists() and probe_out.exists():
                log.info("%s/%s: SKIP (outputs already exist; use --overwrite)", cohort, sample_id)
                n_skipped += 1
                continue

            row = process_sample(source_file, cohort, sample_id, coord_out, probe_out, args.dry_run)
            manifest_rows[(row["dataset_id"], row["cohort"], row["sample_id"])] = row

            if row["status"] in ("ok", "dry_run_ok"):
                n_ok += 1
            else:
                n_failed += 1

            if i % 25 == 0 or i == len(files):
                log.info("cohort=%s progress: %d/%d", cohort, i, len(files))

    if not args.dry_run:
        write_manifest(manifest_path, manifest_rows)
        log.info("Wrote manifest: %s", manifest_path)
    else:
        log.info("Dry run: manifest not written")

    log.info("Done. ok=%d failed=%d skipped=%d", n_ok, n_failed, n_skipped)
    return 1 if n_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
