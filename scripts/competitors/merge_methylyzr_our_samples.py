#!/usr/bin/env python3
"""Merge MethyLYZR predictions (`run_methylyzr_predict.py` output) into the
existing B0/C0/crossNN comparison on our own low-coverage MB/ETMR
samples (`outputs/benchmarks/our_samples/mb_low_coverage_predictions.tsv`,
built by `evaluate_our_samples.py` / `build_sturgeon_bed_our_samples.py`).

Ground truth here is family-level only (MB vs ETMR), from the sample
naming convention with the established MB_591->ETMR override -- same
convention the rest of this comparison uses (see
scripts/data/ingest_our_samples_pileup.py). MethyLYZR's top-1 class is
mapped to that same two-family taxonomy by a simple prefix rule (its
classes are already in "MB, <subtype>" / "ETMR" form, not our 27-code
family_code table, so this is deliberately narrow -- it only resolves the
two families that actually appear in this sample set, anything else would
need the full model_class->family_code taxonomy this session didn't have
access to).

Sturgeon is not included: it has never been run on these samples in this
project (needs a separate python3.9 interpreter, not queued up).

Usage:
    python scripts/competitors/merge_methylyzr_our_samples.py \
        --methylyzr-predictions /tmp/methylyzr_run/methylyzr_predictions.tsv \
        --output outputs/benchmarks/our_samples/mb_low_coverage_predictions_with_methylyzr.tsv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
TRUE_FAMILY = {  # same MB_591->ETMR override as ingest_our_samples_pileup.py
    "MB_591": "ETMR",
}


def methylyzr_class_to_family(cls: str) -> str:
    if cls == "ETMR":
        return "ETMR"
    if cls.startswith("MB"):
        return "MB"
    return cls  # not one of the two families in this sample set


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--existing-predictions", type=Path,
        default=REPO_ROOT / "outputs/benchmarks/our_samples/mb_low_coverage_predictions.tsv",
    )
    ap.add_argument("--methylyzr-predictions", type=Path, required=True)
    ap.add_argument(
        "--output", type=Path,
        default=REPO_ROOT / "outputs/benchmarks/our_samples"
        / "mb_low_coverage_predictions_with_methylyzr.tsv",
    )
    args = ap.parse_args()

    existing = pd.read_csv(args.existing_predictions, sep="\t")
    true_family_by_sample = (
        existing.drop_duplicates("sample_id").set_index("sample_id")["true_family"]
    )

    wide = pd.read_csv(args.methylyzr_predictions, sep="\t")
    class_cols = [c for c in wide.columns if c not in ("sample_id", "number_probes")]

    rows = []
    for _, row in wide.iterrows():
        sample_id = row["sample_id"]
        if sample_id not in true_family_by_sample.index:
            continue
        true_family = TRUE_FAMILY.get(sample_id, true_family_by_sample[sample_id])
        posterior = row[class_cols].astype(float).sort_values(ascending=False)
        top_class = posterior.index[0]
        confidence = posterior.iloc[0]
        predicted_family = methylyzr_class_to_family(top_class)
        # true_family_probability: sum posterior mass over every class that
        # maps to the true family (mirrors aggregate_to_family's semantics).
        true_family_mass = sum(
            p for cls, p in posterior.items() if methylyzr_class_to_family(cls) == true_family
        )
        rows.append({
            "sample_id": sample_id,
            "coverage": "low coverage (~10min ONT)",
            "true_family": true_family,
            "model": "MethyLYZR",
            "n_reads_total": None,
            "predicted_family": predicted_family,
            "confidence": confidence,
            "true_family_probability": true_family_mass,
            "top1_correct": predicted_family == true_family,
            "top3_correct": true_family
            in {methylyzr_class_to_family(c) for c in posterior.index[:3]},
        })

    methylyzr_df = pd.DataFrame(rows)
    merged = pd.concat([existing, methylyzr_df], ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, sep="\t", index=False)
    print(f"Wrote {len(merged)} rows ({len(rows)} new MethyLYZR rows) to {args.output}")

    n_samples = len(rows)
    n_correct = sum(r["top1_correct"] for r in rows)
    print(f"MethyLYZR family top-1 accuracy on {n_samples} scored samples: {n_correct}/{n_samples}")
    missing = sorted(set(true_family_by_sample.index) - set(wide["sample_id"]))
    if missing:
        print(f"Not yet in MethyLYZR predictions (still extracting/pending): {missing}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
