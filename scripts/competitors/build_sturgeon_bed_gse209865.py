#!/usr/bin/env python3
"""Reformat GSE209865's raw per-sample bed.gz files into the tab-delimited
Sturgeon bed format run_sturgeon_predict.py/run_crossnn_predict.py expect.

GSE209865 (registry: role=benchmark_legacy, real nanopore low-pass WGS, one
of this project's own priority test cohorts) ships its raw per-GSM files
already as Sturgeon-shaped rows (chrom, chromStart, chromEnd,
methylation_call, probe_id) -- these are *real* ONT low-pass binary calls
from GEO, not a synthetic threshold-of-beta like build_sturgeon_bed_beta_only.py
produces for the Magibu pools. No decision rule is applied here: the calls
are used exactly as GEO published them. The only change is column order/
whitespace -> tabs (GEO's files are space-aligned, not tab-delimited; the
crossNN reader requires an actual tab-separated file, see
run_crossnn_predict.py's ``sep="\\t"`` read).
"""

from __future__ import annotations

import argparse
import gzip
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from neuromethyl_ont.evaluation.sturgeon_adapter import STURGEON_BED_COLUMNS  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-dir", type=Path, required=True, help="datasets/GSE209865/raw/extracted")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    files = sorted(args.raw_dir.glob("*.bed.gz"))
    if not files:
        ap.error(f"no *.bed.gz under {args.raw_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    n_written = n_skipped = 0
    row_counts = []
    for path in files:
        sample_id = path.name.removesuffix(".bed.gz")
        out_path = args.output_dir / f"{sample_id}.bed"
        if out_path.exists() and not args.overwrite:
            n_skipped += 1
            continue
        with gzip.open(path, "rt") as fh:
            frame = pd.read_csv(fh, sep=r"\s+")
        missing = set(STURGEON_BED_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(f"{path}: missing column(s) {missing}")
        frame[STURGEON_BED_COLUMNS].to_csv(out_path, sep="\t", index=False)
        row_counts.append(len(frame))
        n_written += 1

    print(f"Wrote {n_written} bed file(s), skipped {n_skipped} existing, to {args.output_dir}")
    if row_counts:
        print(f"calls per sample: min={min(row_counts)} median={int(np.median(row_counts))} "
              f"max={max(row_counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
