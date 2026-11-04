#!/usr/bin/env python3
"""Build Sturgeon- or crossNN-format BED inputs from Magibu's immutable
beta-only pools.

This adapter does not manufacture coverage or counts, and must never be
used to evaluate Evidence. Both Sturgeon's and crossNN's input formats accept
exactly one binary methylation call per probe; ``--caller`` picks whose own
decision rule to reproduce: Sturgeon's 0.7/0.3 dead zone (values in the
ambiguous interval omitted), or crossNN's own single 0.6 cutoff, no dead
zone. Earlier revisions of this script only ever produced the Sturgeon bed
and fed it to crossNN too, silently withholding every probe in (0.3, 0.7)
that crossNN's own pipeline would still call.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from neuromethyl_ont.evaluation.crossnn_adapter import (  # noqa: E402
    probe_level_to_crossnn_bed,
)
from neuromethyl_ont.evaluation.sturgeon_adapter import (  # noqa: E402
    probe_level_to_sturgeon_bed,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--magibu-root", type=Path, required=True)
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

    benchmark_root = args.magibu_root / "beta_only"
    manifests = sorted(benchmark_root.glob("*/manifest.tsv"))
    if not manifests:
        ap.error(f"no beta-only manifests under {benchmark_root}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    n_written = n_skipped = 0
    used_counts: list[int] = []
    records: list[dict[str, object]] = []
    seen_sample_ids: set[str] = set()
    for manifest_path in manifests:
        manifest = pd.read_csv(manifest_path, sep="\t")
        required = {"sample_id", "benchmark", "source_file"}
        if not required.issubset(manifest.columns):
            raise ValueError(f"{manifest_path}: missing {sorted(required - set(manifest.columns))}")
        for row in manifest.itertuples(index=False):
            if row.sample_id in seen_sample_ids:
                raise ValueError(f"duplicate Magibu sample_id across benchmarks: {row.sample_id}")
            seen_sample_ids.add(row.sample_id)
            source = Path(row.source_file)
            if not source.exists():
                raise FileNotFoundError(source)
            out_path = args.output_dir / f"{row.sample_id}.bed"
            if out_path.exists() and not args.overwrite:
                n_skipped += 1
                continue
            frame = pd.read_csv(source, sep="\t", usecols=["Probe_ID", "Beta"])
            if frame.Probe_ID.duplicated().any():
                raise ValueError(f"{source}: duplicate Probe_ID")
            probe_frame = frame.rename(
                columns={"Probe_ID": "probe_id", "Beta": "methylation_fraction"}
            )
            if args.caller == "sturgeon":
                bed = probe_level_to_sturgeon_bed(
                    probe_frame,
                    pos_threshold=args.pos_threshold,
                    neg_threshold=args.neg_threshold,
                )
            else:
                bed = probe_level_to_crossnn_bed(probe_frame, threshold=args.threshold)
            bed.to_csv(out_path, sep="\t", index=False)
            n_written += 1
            used_counts.append(len(bed))
            record = {
                "sample_id": row.sample_id,
                "benchmark": row.benchmark,
                "source_file": str(source),
                "n_source_probes": len(frame),
                "n_binary_calls": len(bed),
                "caller": args.caller,
            }
            if args.caller == "sturgeon":
                record["pos_threshold"] = args.pos_threshold
                record["neg_threshold"] = args.neg_threshold
            else:
                record["threshold"] = args.threshold
            records.append(record)

    pd.DataFrame(records).to_csv(args.output_dir / "input_manifest.tsv", sep="\t", index=False)
    print(f"Wrote {n_written} BED file(s), skipped {n_skipped}, to {args.output_dir}")
    if used_counts:
        calls = pd.Series(used_counts)
        print(f"Binary calls: min={calls.min()} median={calls.median():.0f} max={calls.max()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
