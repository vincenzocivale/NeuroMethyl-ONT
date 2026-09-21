#!/usr/bin/env python3
"""Classify one ONT sample with B0, at any coverage -- intra-operative or end-of-run.

B0 encodes each probe as an independent binary call from raw
(methylated_count, unmethylated_count) evidence (docs/B0_PRODUCTION_MODEL.md).
There is no depth-dependent step: calling this script on a partial,
low-coverage intra-operative pileup and calling it again once the run has
finished (higher coverage, more probes called) uses the identical model and
code path -- only the amount of available evidence differs.

Input is a per-probe TSV with columns ``probe_id``, ``methylated_count``,
``unmethylated_count`` (the canonical Rapid-CNS2/ONT-WGS schema,
docs/RAPIDCNS2_CANONICAL_PREPROCESSING.md), already probe-level aggregated
(summed across duplicate rows). Probes absent from the file, or from the
training feature index, are treated as unobserved (contribute 0 to B0).

Usage:
    python scripts/inference/classify_sample.py \\
        --feature-index "$NEUROMETHYL_DATA_ROOT/derived/reference/GSE90496/feature_index.json" \\
        --checkpoint outputs/experiments/b0_fixed80_seed17_bs128_w16_opt/checkpoint_best.pt \\
        --sample-counts PATH/sample_counts.tsv \\
        --sample-id patient001_intraop \\
        --output outputs/inference/patient001_intraop.json
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
from neuromethyl_ont.data.sequencing_simulator import binary_transform  # noqa: E402
from neuromethyl_ont.evaluation.explainability import explain_prediction  # noqa: E402
from neuromethyl_ont.models.evidence_linear import build_linear_classifier  # noqa: E402


def load_counts_aligned(path: Path, probe_ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Read a probe-level counts TSV and reindex it to ``probe_ids`` order.

    Probes missing from the file get ``m = u = 0`` (unobserved, matching
    ``binary_transform``'s ``n == 0`` convention).
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


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--feature-index", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--sample-counts", type=Path, required=True)
    ap.add_argument("--sample-id", default="sample")
    ap.add_argument("--top-n-loci", type=int, default=20)
    ap.add_argument("--n-runner-up", type=int, default=2)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    feature_index = load_feature_index(args.feature_index)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    representation = checkpoint["representation"]
    if representation != "binary":
        raise ValueError(f"expected a B0 checkpoint, got representation={representation!r}")

    class_names = [
        name for name, _ in sorted(checkpoint["class_to_idx"].items(), key=lambda kv: kv[1])
    ]
    model = build_linear_classifier(
        n_features=checkpoint["n_features"], n_classes=len(class_names)
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    m, u = load_counts_aligned(args.sample_counts, feature_index.probe_ids)
    x = binary_transform(m, u)
    n_observed = int((m + u > 0).sum())

    with torch.no_grad():
        logits = model(torch.as_tensor(x, dtype=torch.float32).unsqueeze(0))[0]
    probs = torch.softmax(logits, dim=0).numpy()
    ranked = np.argsort(-probs)
    top3 = [
        {"class_name": class_names[i], "probability": float(probs[i])} for i in ranked[:3]
    ]

    report = explain_prediction(
        model,
        x,
        feature_index.probe_ids,
        class_names,
        sample_id=args.sample_id,
        top_n=args.top_n_loci,
        n_runner_up=args.n_runner_up,
    )

    output = {
        "sample_id": args.sample_id,
        "n_probes_in_reference": feature_index.n_features,
        "n_probes_observed": n_observed,
        "top3": top3,
        "explainability": {
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
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    print(
        f"{args.sample_id}: {n_observed}/{feature_index.n_features} probes observed -> "
        f"top1={top3[0]['class_name']} (p={top3[0]['probability']:.3f}); "
        f"wrote {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
