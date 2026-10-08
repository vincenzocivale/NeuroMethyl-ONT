#!/usr/bin/env python3
"""Convert canonical probe-level parquet (Rapid-CNS2/ONT-WGS/GSE289246) into
one binary bed file per sample, in either Sturgeon's own decision rule
(0.7/0.3 dead zone) or crossNN's own decision rule (single 0.6 cutoff, no
dead zone) -- see ``--caller`` and
src/neuromethyl_ont/evaluation/{sturgeon_adapter,crossnn_adapter}.py for the
two rules. Earlier revisions of this script only ever produced the Sturgeon
bed and fed it to crossNN too, which silently withholds every probe in the
(0.3, 0.7) range that crossNN's own pipeline would still call -- use
``--caller crossnn`` for crossNN's own input from now on, and re-point
``run_crossnn_predict.py --bed-dir`` at that output instead of Sturgeon's.

Usage:
    python scripts/competitors/build_sturgeon_bed_inputs.py \\
        --probe-level-dir \\
            "$NEUROMETHYL_DATA_ROOT/derived/array_aligned/probe_level/RapidCNS2/Rapid_CNS2" \\
        --output-dir outputs/benchmarks/sturgeon/rapidcns2_bed \\
        --caller sturgeon [--pos-threshold 0.7] [--neg-threshold 0.3]

    python scripts/competitors/build_sturgeon_bed_inputs.py \\
        --probe-level-dir ... --output-dir outputs/benchmarks/crossnn/rapidcns2_bed \\
        --caller crossnn [--threshold 0.6]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from neuromethyl_ont.data.canonical import (  # noqa: E402
    aggregate_to_probe_level,
    to_coordinate_level,
)
from neuromethyl_ont.data.gse289246 import read_gse289246_sample  # noqa: E402
from neuromethyl_ont.evaluation.crossnn_adapter import (  # noqa: E402
    probe_level_to_crossnn_bed,
)
from neuromethyl_ont.evaluation.sturgeon_adapter import (  # noqa: E402
    probe_level_to_sturgeon_bed,
)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--probe-level-dir", type=Path,
        help="directory of <sample_id>.parquet canonical probe-level files",
    )
    source.add_argument(
        "--gse289246-raw-dir", type=Path,
        help="directory of GSE289246 *.CG.bed.gz/*.CpG.bed.gz raw bedMethyl files",
    )
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument(
        "--caller", choices=["sturgeon", "crossnn"], default="sturgeon",
        help="which competitor's own decision rule to reproduce (default: sturgeon, "
        "for backward compatibility -- pass crossnn to build crossNN's own input)",
    )
    ap.add_argument("--pos-threshold", type=float, default=0.7, help="sturgeon only")
    ap.add_argument("--neg-threshold", type=float, default=0.3, help="sturgeon only")
    ap.add_argument("--threshold", type=float, default=0.6, help="crossnn only")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    def to_bed(probe_df: pd.DataFrame) -> pd.DataFrame:
        if args.caller == "sturgeon":
            return probe_level_to_sturgeon_bed(
                probe_df, pos_threshold=args.pos_threshold, neg_threshold=args.neg_threshold
            )
        return probe_level_to_crossnn_bed(probe_df, threshold=args.threshold)

    files = (
        sorted(args.probe_level_dir.glob("*.parquet"))
        if args.probe_level_dir is not None
        else sorted(args.gse289246_raw_dir.glob("*.bed.gz"))
    )
    if not files:
        source_dir = args.probe_level_dir or args.gse289246_raw_dir
        print(f"no input files found under {source_dir}", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)

    n_written = n_skipped = 0
    n_probes_used = []
    for f in files:
        sample_id = f.name.split(".")[0]
        out_path = args.output_dir / f"{sample_id}.bed"
        if out_path.exists() and not args.overwrite:
            n_skipped += 1
            continue

        if args.probe_level_dir is not None:
            probe_df = pd.read_parquet(f)
        else:
            coordinate_df = to_coordinate_level(read_gse289246_sample(f), sample_id)
            probe_df = aggregate_to_probe_level(coordinate_df)
        bed = to_bed(probe_df)
        bed.to_csv(out_path, sep="\t", index=False)
        n_written += 1
        n_probes_used.append(len(bed))
        if n_written % 25 == 0 or n_written == len(files):
            print(f"progress: {n_written}/{len(files)}")

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
