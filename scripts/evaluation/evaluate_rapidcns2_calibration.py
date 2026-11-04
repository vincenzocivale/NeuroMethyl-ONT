#!/usr/bin/env python3
"""Binary calibration and top-3 accuracy on Rapid_CNS2, at full 91-class
resolution (not the family-level aggregation evaluate_rapidcns2.py
uses for cross-model comparison). Real per-patient coverage (median ~2x,
not pooled, not synthetic), ground truth from Supplementary Table 1
(`load_rapidcns2_labels` -- rows with a family-only/unresolved true class
are dropped, not guessed, same policy as evaluate_rapidcns2.py).

Writes the same {calibration,top3}.tsv shape evaluate_gse109379.py does, so
plot_b0_calibration_top3.py can draw either cohort with the same code.

Usage:
    python scripts/evaluation/evaluate_rapidcns2_calibration.py \
        --reference-dir $NEUROMETHYL_DATA_ROOT/derived/reference/GSE90496 \
        --binary-checkpoint outputs/experiments/b0_fixed80_seed17_bs128_w16_opt/checkpoint_best.pt \
        --rapid-dir $NEUROMETHYL_DATA_ROOT/derived/array_aligned/probe_level/RapidCNS2/Rapid_CNS2 \
        --rapid-labels-xlsx \
            $NEUROMETHYL_DATA_ROOT/datasets/RapidCNS2/metadata/Supplementary_Tables_1_2.xlsx \
        --output-dir outputs/benchmarks/rapidcns2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from neuromethyl_ont.data.array_reference import load_feature_index  # noqa: E402
from neuromethyl_ont.data.rapidcns2_labels import load_rapidcns2_labels  # noqa: E402
from neuromethyl_ont.evaluation.classification import (  # noqa: E402
    classification_metrics,
    expected_calibration_error,
    negative_log_likelihood,
    reliability_bins,
    softmax_probs,
    top_k_accuracy,
    top_k_table,
)
from neuromethyl_ont.evaluation.cohort_scoring import (  # noqa: E402
    extract_features,
    load_torch_checkpoint,
)
from neuromethyl_ont.training.trainer import predict_logits  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--reference-dir", type=Path, required=True)
    ap.add_argument("--binary-checkpoint", "--b0-checkpoint", type=Path, required=True)
    ap.add_argument("--rapid-dir", type=Path, required=True)
    ap.add_argument("--rapid-labels-xlsx", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    feature_index = load_feature_index(args.reference_dir / "feature_index.json")
    mu = (
        pd.read_parquet(args.reference_dir / "probe_statistics.parquet")
        .set_index("probe_id")["mean_beta"]
        .reindex(feature_index.probe_ids)
        .to_numpy()
    )
    feature_pos = {probe: i for i, probe in enumerate(feature_index.probe_ids)}

    b0_model, b0_classes = load_torch_checkpoint(args.binary_checkpoint)
    class_to_idx = {c: i for i, c in enumerate(b0_classes)}

    rapid_files = sorted(args.rapid_dir.glob("*.parquet"))
    print(f"Rapid_CNS2: {len(rapid_files)} files", flush=True)
    features = extract_features(rapid_files, feature_pos, mu)
    b0_probs_all = softmax_probs(predict_logits(b0_model, features["binary"]))

    labels = load_rapidcns2_labels(args.rapid_labels_xlsx, set(b0_classes)).labels
    label_by_sample = dict(zip(labels.sample_id, labels.label, strict=True))
    row_pos = {sample: i for i, sample in enumerate(features["sample_id"])}

    sample_ids, y_true, rows = [], [], []
    for sample_id, label in label_by_sample.items():
        row = row_pos.get(sample_id)
        if row is None:
            continue
        sample_ids.append(sample_id)
        y_true.append(class_to_idx[label])
        rows.append(row)
    y_true = np.array(y_true, dtype=np.int64)
    b0_probs = b0_probs_all[np.array(rows, dtype=np.int64)]
    print(f"Scoring {len(sample_ids)} labelled samples (full 91-class truth)", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    prediction = b0_probs.argmax(axis=1)
    metrics = {
        "model": "Binary",
        "n_samples": len(sample_ids),
        **classification_metrics(y_true, prediction),
        "top3_accuracy": top_k_accuracy(b0_probs, y_true, k=3),
        "nll": negative_log_likelihood(b0_probs, y_true),
        "ece": expected_calibration_error(b0_probs, y_true),
    }
    pd.DataFrame([metrics]).to_csv(
        args.output_dir / "rapidcns2_class_metrics.tsv", sep="\t", index=False
    )

    bins = reliability_bins(b0_probs, y_true, n_bins=10)
    pd.DataFrame(bins).to_csv(
        args.output_dir / "rapidcns2_b0_calibration.tsv", sep="\t", index=False
    )

    top3 = top_k_table(b0_probs, y_true, b0_classes, k=3)
    top3_frame = pd.DataFrame(top3)
    top3_frame.insert(0, "sample_id", sample_ids)
    top3_frame.to_csv(args.output_dir / "rapidcns2_b0_top3.tsv", sep="\t", index=False)

    print("\n=== Rapid_CNS2, Binary, full 91-class ===")
    print(pd.DataFrame([metrics]).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
