#!/usr/bin/env python3
"""Convert canonical probe-level parquet (Rapid-CNS2/ONT-WGS/GSE289246) into
one Sturgeon-format bed file per sample (scripts/benchmarks/run_sturgeon_predict.py's
input; see src/neuromethyl_ont/evaluation/sturgeon_adapter.py for the format
and the decision rule it reproduces).

Usage:
    python scripts/benchmarks/build_sturgeon_bed_inputs.py \\
        --probe-level-dir \\
            "$NEUROMETHYL_DATA_ROOT/derived/array_aligned/probe_level/RapidCNS2/Rapid_CNS2" \\
        --output-dir outputs/benchmarks/sturgeon/rapidcns2_bed \\
        [--pos-threshold 0.7] [--neg-threshold 0.3]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from neuromethyl_ont.evaluation.sturgeon_adapter import (  # noqa: E402
    probe_level_to_sturgeon_bed,
)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--probe-level-dir", type=Path, required=True,
        help="directory of <sample_id>.parquet canonical probe-level files",
    )
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--pos-threshold", type=float, default=0.7)
    ap.add_argument("--neg-threshold", type=float, default=0.3)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    files = sorted(args.probe_level_dir.glob("*.parquet"))
    if not files:
        print(f"no *.parquet files found under {args.probe_level_dir}", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)

    n_written = n_skipped = 0
    n_probes_used = []
    for f in files:
        sample_id = f.stem
        out_path = args.output_dir / f"{sample_id}.bed"
        if out_path.exists() and not args.overwrite:
            n_skipped += 1
            continue

        probe_df = pd.read_parquet(f)
        bed = probe_level_to_sturgeon_bed(
            probe_df, pos_threshold=args.pos_threshold, neg_threshold=args.neg_threshold
        )
        bed.to_csv(out_path, sep="\t", index=False)
        n_written += 1
        n_probes_used.append(len(bed))

    print(f"Wrote {n_written} bed file(s) to {args.output_dir} (skipped {n_skipped} existing)")
    if n_probes_used:
        s = pd.Series(n_probes_used)
        print(
            f"Probes used per sample: min={s.min()} median={s.median():.0f} max={s.max()} "
            f"(of 428799 GSE90496 probes)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
