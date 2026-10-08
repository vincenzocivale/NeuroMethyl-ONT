#!/usr/bin/env python3
"""B0 vs C0 on our own real samples (`scripts/data/ingest_our_samples_pileup.py`
output): ETMR and MB, matched high/low coverage where available. Ground
truth here is family-level only, from the sample naming convention (no
per-subclass label was found for these samples anywhere in the source
data -- see project memory) -- so this reports whether each model's top
prediction (and top-3) lands on the right *family*, not full family_metrics
against a diverse label set the way evaluate_rapidcns2.py does.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from neuromethyl_ont.data.sequencing_simulator import (  # noqa: E402
    binary_transform,
    continuous_transform,
)
from neuromethyl_ont.evaluation.classification import softmax_probs  # noqa: E402
from neuromethyl_ont.evaluation.cohort_scoring import load_torch_checkpoint  # noqa: E402
from neuromethyl_ont.evaluation.family_taxonomy import aggregate_to_family  # noqa: E402
from neuromethyl_ont.training.trainer import predict_logits  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--counts-dir", type=Path, required=True, help="ingest_our_samples_pileup.py output"
    )
    ap.add_argument("--b0-checkpoint", type=Path, required=True)
    ap.add_argument("--c0-checkpoint", type=Path, required=True)
    ap.add_argument("--reference-dir", type=Path, required=True)
    ap.add_argument("--taxonomy", type=Path, required=True)
    ap.add_argument("--coverage-label", required=True, help="e.g. 'low coverage (~10min)'")
    ap.add_argument(
        "--crossnn-predictions", type=Path, default=None,
        help="optional: scripts/competitors/run_crossnn_predict.py output for this cohort",
    )
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    m = np.load(args.counts_dir / "m.npy")
    u = np.load(args.counts_dir / "u.npy")
    manifest = pd.read_csv(args.counts_dir / "manifest.tsv", sep="\t")
    mu = (
        pd.read_parquet(args.reference_dir / "probe_statistics.parquet")["mean_beta"].to_numpy()
    )

    taxonomy = pd.read_csv(args.taxonomy, sep="\t")
    family_by_class = {
        str(row.model_class): str(row.family_code).strip()
        for row in taxonomy.itertuples(index=False)
        if pd.notna(row.family_code) and str(row.family_code).strip()
    }

    b0_model, b0_classes = load_torch_checkpoint(args.b0_checkpoint)
    c0_model, c0_classes = load_torch_checkpoint(args.c0_checkpoint)

    b0_x = binary_transform(m, u, threshold=0.6)
    b0_probs = softmax_probs(predict_logits(b0_model, b0_x))
    b0_fam, families = aggregate_to_family(b0_probs, b0_classes, family_by_class)

    c0_x = continuous_transform(m, u, mu)
    c0_probs = softmax_probs(predict_logits(c0_model, c0_x))
    c0_fam, c0_families = aggregate_to_family(c0_probs, c0_classes, family_by_class)
    if c0_families != families:
        raise RuntimeError("B0/C0 family layouts disagree")

    model_families = [("B0", b0_fam), ("C0", c0_fam)]
    if args.crossnn_predictions is not None:
        table = pd.read_csv(args.crossnn_predictions, sep="\t")
        score_columns = [c for c in table.columns if c not in {"sample_id", "number_probes"}]
        crossnn_probs, crossnn_families = aggregate_to_family(
            table[score_columns].to_numpy(dtype=np.float64), score_columns, family_by_class
        )
        by_sample = dict(zip(table.sample_id, crossnn_probs, strict=True))
        missing = set(manifest.sample_id) - set(by_sample)
        if missing:
            raise RuntimeError(f"crossNN predictions missing for: {sorted(missing)}")
        stacked = np.stack([by_sample[s] for s in manifest.sample_id])
        crossnn_fam = np.zeros((len(manifest), len(families)), dtype=np.float64)
        for col, family in enumerate(crossnn_families):
            crossnn_fam[:, families.index(family)] = stacked[:, col]
        model_families.append(("crossNN", crossnn_fam))

    rows = []
    for model_name, fam_probs in model_families:
        order = np.argsort(-fam_probs, axis=1)
        for i, sample_id in enumerate(manifest.sample_id):
            true_family = manifest.family_code.iloc[i]
            top3 = [families[j] for j in order[i, :3]]
            rows.append(
                {
                    "sample_id": sample_id,
                    "coverage": args.coverage_label,
                    "true_family": true_family,
                    "model": model_name,
                    "n_reads_total": int(manifest.n_reads_total.iloc[i]),
                    "predicted_family": families[fam_probs[i].argmax()],
                    "confidence": float(fam_probs[i].max()),
                    "true_family_probability": float(
                        fam_probs[i, families.index(true_family)]
                    ),
                    "top1_correct": families[fam_probs[i].argmax()] == true_family,
                    "top3_correct": true_family in top3,
                }
            )
    result = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, sep="\t", index=False)

    summary = result.groupby("model")[["top1_correct", "top3_correct"]].mean()
    print(f"\n=== our samples, {args.coverage_label}, n={manifest.sample_id.nunique()} ===")
    print(summary.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
