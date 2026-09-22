#!/usr/bin/env python3
"""Classify one ONT sample with the B0+E1 ensemble, at any coverage.

Averages B0's (binary, threshold-based) and E1's (coverage-shrunk continuous)
softmax probabilities. Empirically (docs/B0_PRODUCTION_MODEL.md), this beats
either model alone on the one real cohort where both were tested end-to-end:
matches B0's 100% on GSE289246 (real end-of-run coverage, where E1 alone
drops to 77.8% due to a platform batch-effect vulnerability) while matching
E1's accuracy gain over B0 on Rapid-CNS2 (real low/medium coverage) and
improving calibration (NLL) beyond both individual models. There is no
depth-dependent branch in either model or its encoding: calling this script
on a partial intra-operative pileup and again once the run has finished uses
the identical models/checkpoints -- only how much evidence is available
changes.

Input is a per-probe TSV with columns ``probe_id``, ``methylated_count``,
``unmethylated_count`` (the canonical Rapid-CNS2/ONT-WGS schema,
docs/RAPIDCNS2_CANONICAL_PREPROCESSING.md), already probe-level aggregated
(summed across duplicate rows). Probes absent from the file, or from the
training feature index, are treated as unobserved.

Usage:
    python scripts/inference/classify_sample.py \\
        --feature-index "$NEUROMETHYL_DATA_ROOT/derived/reference/GSE90496/feature_index.json" \\
        --reference-dir "$NEUROMETHYL_DATA_ROOT/derived/reference/GSE90496" \\
        --b0-checkpoint outputs/experiments/b0_fixed80_seed17_bs128_w16_opt/checkpoint_best.pt \\
        --e1-checkpoint outputs/experiments/e1_kappa2_fixed80_seed17_bs128_w2/checkpoint_best.pt \\
        --sample-counts PATH/sample_counts.tsv \\
        --sample-id patient001 \\
        --output outputs/inference/patient001.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from neuromethyl_ont.data.array_reference import load_feature_index  # noqa: E402
from neuromethyl_ont.data.sequencing_simulator import (  # noqa: E402
    binary_transform,
    evidence_transform,
)
from neuromethyl_ont.evaluation.explainability import explain_prediction  # noqa: E402
from neuromethyl_ont.models.evidence_linear import build_linear_classifier  # noqa: E402


def load_counts_aligned(path: Path, probe_ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Read a probe-level counts TSV and reindex it to ``probe_ids`` order.

    Probes missing from the file get ``m = u = 0`` (unobserved, matching
    ``binary_transform``/``evidence_transform``'s ``n == 0`` convention).
    """
    table = pd.read_csv(path, sep="\t")
    required = {"probe_id", "methylated_count", "unmethylated_count"}
    missing_cols = required - set(table.columns)
    if missing_cols:
        raise ValueError(f"{path} is missing columns: {sorted(missing_cols)}")

    by_probe = table.set_index("probe_id")[["methylated_count", "unmethylated_count"]]
    m = np.zeros(len(probe_ids), dtype=np.float64)
    u = np.zeros(len(probe_ids), dtype=np.float64)
    for i, probe_id in enumerate(probe_ids):
        if probe_id in by_probe.index:
            row = by_probe.loc[probe_id]
            m[i], u[i] = float(row["methylated_count"]), float(row["unmethylated_count"])
    return m, u


def load_linear_checkpoint(path: Path, expected_representation: str):
    checkpoint = torch.load(path, map_location="cpu")
    representation = checkpoint["representation"]
    if representation != expected_representation:
        raise ValueError(
            f"{path}: expected representation={expected_representation!r}, "
            f"got {representation!r}"
        )
    class_names = [
        name for name, _ in sorted(checkpoint["class_to_idx"].items(), key=lambda kv: kv[1])
    ]
    model = build_linear_classifier(
        n_features=checkpoint["n_features"], n_classes=len(class_names)
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, class_names


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--feature-index", type=Path, required=True)
    ap.add_argument("--reference-dir", type=Path, required=True, help="for E1's per-probe mean")
    ap.add_argument("--b0-checkpoint", type=Path, required=True)
    ap.add_argument("--e1-checkpoint", type=Path, required=True)
    ap.add_argument("--e1-kappa", type=float, default=2.0)
    ap.add_argument("--sample-counts", type=Path, required=True)
    ap.add_argument("--sample-id", default="sample")
    ap.add_argument("--top-n-loci", type=int, default=20)
    ap.add_argument("--n-runner-up", type=int, default=2)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    feature_index = load_feature_index(args.feature_index)
    mu = (
        pd.read_parquet(args.reference_dir / "probe_statistics.parquet")
        .set_index("probe_id")["mean_beta"]
        .reindex(feature_index.probe_ids)
        .to_numpy()
    )

    b0_model, class_names = load_linear_checkpoint(args.b0_checkpoint, "binary")
    e1_model, e1_class_names = load_linear_checkpoint(args.e1_checkpoint, "evidence")
    if e1_class_names != class_names:
        raise ValueError("B0 and E1 checkpoints must cover exactly the same classes, in order")

    m, u = load_counts_aligned(args.sample_counts, feature_index.probe_ids)
    n_observed = int((m + u > 0).sum())

    x_b0 = binary_transform(m, u)
    x_e1 = evidence_transform(m, u, mu, args.e1_kappa)

    with torch.no_grad():
        b0_probs = torch.softmax(
            b0_model(torch.as_tensor(x_b0, dtype=torch.float32).unsqueeze(0))[0], dim=0
        ).numpy()
        e1_probs = torch.softmax(
            e1_model(torch.as_tensor(x_e1, dtype=torch.float32).unsqueeze(0))[0], dim=0
        ).numpy()
    ensemble_probs = 0.5 * b0_probs + 0.5 * e1_probs

    ranked = np.argsort(-ensemble_probs)
    top3 = [
        {"class_name": class_names[i], "probability": float(ensemble_probs[i])}
        for i in ranked[:3]
    ]

    b0_report = explain_prediction(
        b0_model, x_b0, feature_index.probe_ids, class_names,
        sample_id=args.sample_id, top_n=args.top_n_loci, n_runner_up=args.n_runner_up,
    )
    e1_report = explain_prediction(
        e1_model, x_e1, feature_index.probe_ids, class_names,
        sample_id=args.sample_id, top_n=args.top_n_loci, n_runner_up=args.n_runner_up,
    )

    def _serialize(report):
        return {
            class_name: [
                {
                    "probe_id": c.probe_id,
                    "feature_value": c.feature_value,
                    "weight": c.weight,
                    "contribution": c.contribution,
                }
                for c in contributions
            ]
            for class_name, contributions in report.contributions_by_class.items()
        }

    output = {
        "sample_id": args.sample_id,
        "n_probes_in_reference": feature_index.n_features,
        "n_probes_observed": n_observed,
        "top3": top3,
        "per_model": {
            "B0": {
                "probability_of_top1": float(b0_probs[ranked[0]]),
                "explainability": _serialize(b0_report),
            },
            "E1": {
                "probability_of_top1": float(e1_probs[ranked[0]]),
                "explainability": _serialize(e1_report),
            },
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    print(
        f"{args.sample_id}: {n_observed}/{feature_index.n_features} probes observed -> "
        f"top1={top3[0]['class_name']} (ensemble p={top3[0]['probability']:.3f}, "
        f"B0 p={b0_probs[ranked[0]]:.3f}, E1 p={e1_probs[ranked[0]]:.3f}); wrote {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
