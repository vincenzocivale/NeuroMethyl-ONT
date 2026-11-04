#!/usr/bin/env python3
"""Build Sturgeon- or crossNN-format BED inputs from our own samples'
pileup-derived counts (`scripts/data/ingest_our_samples_pileup.py` output:
m.npy/u.npy/probe_ids.tsv/manifest.tsv). Needed to run Sturgeon/crossNN on
the same evidence Binary/Beta already score in
`scripts/evaluation/evaluate_our_samples.py` -- neither had been run on
this cohort before (a previously open gap).

``--caller`` picks which competitor's own decision rule to reproduce:
Sturgeon's 0.7/0.3 dead zone, or crossNN's own single 0.6 cutoff (no dead
zone) -- see src/neuromethyl_ont/evaluation/{sturgeon_adapter,crossnn_adapter}.py.
Earlier revisions of this script only ever produced the Sturgeon bed and
fed it to crossNN too, silently withholding every probe in (0.3, 0.7) that
crossNN's own pipeline would still call.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from neuromethyl_ont.evaluation.crossnn_adapter import probe_level_to_crossnn_bed  # noqa: E402
from neuromethyl_ont.evaluation.sturgeon_adapter import probe_level_to_sturgeon_bed  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--counts-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument(
        "--caller", choices=["sturgeon", "crossnn"], default="sturgeon",
        help="which competitor's own decision rule to reproduce (default: sturgeon, "
        "for backward compatibility -- pass crossnn to build crossNN's own input)",
    )
    ap.add_argument("--pos-threshold", type=float, default=0.7, help="sturgeon only")
    ap.add_argument("--neg-threshold", type=float, default=0.3, help="sturgeon only")
    ap.add_argument("--threshold", type=float, default=0.6, help="crossnn only")
    args = ap.parse_args()

    m = np.load(args.counts_dir / "m.npy")
    u = np.load(args.counts_dir / "u.npy")
    probe_ids = pd.read_csv(args.counts_dir / "probe_ids.tsv", sep="\t")["probe_id"].to_numpy()
    manifest = pd.read_csv(args.counts_dir / "manifest.tsv", sep="\t")

    n = m + u
    with np.errstate(invalid="ignore", divide="ignore"):
        beta = np.where(n > 0, m / n, np.nan)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for row, sample_id in enumerate(manifest.sample_id):
        probe_df = pd.DataFrame({"probe_id": probe_ids, "methylation_fraction": beta[row]})
        if args.caller == "sturgeon":
            bed = probe_level_to_sturgeon_bed(
                probe_df, pos_threshold=args.pos_threshold, neg_threshold=args.neg_threshold
            )
        else:
            bed = probe_level_to_crossnn_bed(probe_df, threshold=args.threshold)
        bed.to_csv(args.output_dir / f"{sample_id}.bed", sep="\t", index=False)
        print(f"{sample_id}: {len(bed)} binary calls", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
