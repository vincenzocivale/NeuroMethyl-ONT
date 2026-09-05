#!/usr/bin/env python3
"""Validate an IDAT->beta reconstruction (scripts/data/idat_to_beta_mnp.R)
against the intact rows of a known-malformed GEO processed matrix.

GSE109379_processed_data.txt.gz is truncated mid-download (see
docs/GSE109379_IDAT_RECONSTRUCTION.md), but its intact ~42k rows are still
real, independently-computed beta values -- comparing against them is the
correctness check for the from-scratch MNPpreprocessIllumina reconstruction,
without trusting either source blindly.

Usage:
    python scripts/data/validate_idat_reconstruction.py \\
        --reconstructed-dir PATH --manifest PATH --geo-matrix PATH \\
        [--min-pearson-r 0.98] [--max-median-abs-diff 0.02]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from neuromethyl_ont.data.geo_series import read_geo_beta_matrix_tolerant  # noqa: E402
from neuromethyl_ont.data.idat_reconstruction import (  # noqa: E402
    compare_beta_matrices,
    load_mnp_preprocessed,
)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--reconstructed-dir", type=Path, required=True, help="idat_to_beta_mnp.R output dir"
    )
    ap.add_argument(
        "--manifest", type=Path, required=True, help="export_idat_sample_manifest.py output"
    )
    ap.add_argument(
        "--geo-matrix", type=Path, required=True, help="the malformed GEO processed matrix"
    )
    ap.add_argument("--min-pearson-r", type=float, default=0.98)
    ap.add_argument("--max-median-abs-diff", type=float, default=0.02)
    ap.add_argument("--report", type=Path, default=None)
    args = ap.parse_args()

    reconstructed = load_mnp_preprocessed(args.reconstructed_dir)
    manifest = pd.read_csv(args.manifest, sep="\t")
    manifest_by_gsm = manifest.set_index("gsm")

    missing = [gsm for gsm in reconstructed.sample_ids if gsm not in manifest_by_gsm.index]
    if missing:
        print(
            f"error: {len(missing)} reconstructed sample(s) not in manifest: {missing[:5]}",
            file=sys.stderr,
        )
        return 2
    sample_columns = manifest_by_gsm.loc[reconstructed.sample_ids, "sample_column"].tolist()

    print(f"Reading GEO matrix (tolerant of malformed rows): {args.geo_matrix}")
    geo_beta, geo_probe_ids, n_skipped = read_geo_beta_matrix_tolerant(
        args.geo_matrix, sample_columns=sample_columns
    )
    print(
        f"GEO matrix: {geo_beta.shape[0]} samples x {geo_beta.shape[1]} intact probes "
        f"({n_skipped} row(s) skipped)"
    )

    result = compare_beta_matrices(
        reconstructed.beta, reconstructed.probe_ids, geo_beta, geo_probe_ids
    )
    result["n_samples"] = len(reconstructed.sample_ids)
    result["n_geo_rows_skipped"] = n_skipped
    print(json.dumps(result, indent=2))

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2))

    passed = (
        result["pearson_r"] >= args.min_pearson_r
        and result["median_abs_diff"] <= args.max_median_abs_diff
    )
    print("PASS" if passed else "FAIL", "-- reconstruction vs. intact GEO rows")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
