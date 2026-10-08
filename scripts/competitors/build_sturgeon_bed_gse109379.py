#!/usr/bin/env python3
"""Build Sturgeon or crossNN BED inputs from GSE109379's IDAT-reconstructed
beta matrix (scripts/data/idat_to_beta_mnp.R output), one file per sample.

Same threshold adapters as build_sturgeon_bed_beta_only.py (Magibu pools):
``--caller`` picks whose own decision rule to reproduce -- Sturgeon's
0.7/0.3 dead zone (probe_level_to_sturgeon_bed, ambiguous values omitted)
or crossNN's own single 0.6 cutoff, no dead zone
(probe_level_to_crossnn_bed). Earlier revisions of this script only ever
produced the Sturgeon bed and fed it to crossNN too, silently withholding
every probe in (0.3, 0.7) that crossNN's own pipeline would still call.
Uses the full IDAT-reconstructed probe set (485512 probes), not the
GSE90496 feature-index subset used for B0/C0 -- Sturgeon/crossNN each
subset to their own probe panel internally, so there is no reason to
pre-restrict here.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from neuromethyl_ont.data.idat_reconstruction import load_mnp_preprocessed  # noqa: E402
from neuromethyl_ont.evaluation.crossnn_adapter import probe_level_to_crossnn_bed  # noqa: E402
from neuromethyl_ont.evaluation.sturgeon_adapter import probe_level_to_sturgeon_bed  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mnp-dir", type=Path, required=True)
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

    print(f"Loading MNP-reconstructed beta from {args.mnp_dir} ...", flush=True)
    mnp = load_mnp_preprocessed(args.mnp_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    probe_ids = pd.Index(mnp.probe_ids)

    n_written = n_skipped = 0
    used_counts: list[int] = []
    for row, sample_id in enumerate(mnp.sample_ids):
        out_path = args.output_dir / f"{sample_id}.bed"
        if out_path.exists() and not args.overwrite:
            n_skipped += 1
            continue
        beta = mnp.beta[row]
        valid = ~np.isnan(beta)
        frame = pd.DataFrame(
            {"probe_id": probe_ids[valid], "methylation_fraction": beta[valid]}
        )
        if args.caller == "sturgeon":
            bed = probe_level_to_sturgeon_bed(
                frame, pos_threshold=args.pos_threshold, neg_threshold=args.neg_threshold
            )
        else:
            bed = probe_level_to_crossnn_bed(frame, threshold=args.threshold)
        bed.to_csv(out_path, sep="\t", index=False)
        used_counts.append(len(bed))
        n_written += 1
        if n_written % 100 == 0:
            print(f"progress: {n_written}/{len(mnp.sample_ids)}", flush=True)

    print(f"Wrote {n_written} bed file(s), skipped {n_skipped} existing, to {args.output_dir}")
    if used_counts:
        print(f"probes per bed: min={min(used_counts)} median={int(np.median(used_counts))} "
              f"max={max(used_counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
