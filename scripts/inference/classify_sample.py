#!/usr/bin/env python3
"""Classify one ONT sample with the B0+E1 ensemble, at any coverage.

Averages B0's (binary, threshold-based) and E1's (coverage-shrunk continuous)
softmax probabilities. Empirically (docs/PRODUCTION_MODEL.md), this beats
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

Optional two-stage mode (CNS + acute leukemia): pass ``--router-checkpoint``
plus the ``--alma-*`` arguments. A domain router first decides between the CNS
head and the ALMA leukemia head from the same counts; below
``--router-threshold`` the sample is reported as ``undetermined`` (both heads'
top-3 are still listed, labelled by head, for transparency).

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

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from neuromethyl_ont.evaluation.explainability import explain_prediction  # noqa: E402
from neuromethyl_ont.inference.two_stage import (  # noqa: E402
    UNDETERMINED,
    decide_domain,
    load_counts_aligned,
    load_head,
    load_router,
    route,
    score_head,
)
from neuromethyl_ont.training.router import DOMAINS  # noqa: E402


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


def classify_with_head(head, counts_path: Path, sample_id: str, top_n: int, n_runner_up: int):
    m, u = load_counts_aligned(counts_path, head.feature_index.probe_ids)
    scores = score_head(head, m, u)
    ranked = np.argsort(-scores.ensemble_probs)
    top3 = [
        {"class_name": head.class_names[i], "probability": float(scores.ensemble_probs[i])}
        for i in ranked[:3]
    ]
    per_model = {}
    for name, model, x, probs in (
        ("B0", head.b0_model, scores.x_b0, scores.b0_probs),
        ("E1", head.e1_model, scores.x_e1, scores.e1_probs),
    ):
        report = explain_prediction(
            model, x, head.feature_index.probe_ids, head.class_names,
            sample_id=sample_id, top_n=top_n, n_runner_up=n_runner_up,
        )
        per_model[name] = {
            "probability_of_top1": float(probs[ranked[0]]),
            "explainability": _serialize(report),
        }
    return {
        "head": head.name,
        "n_probes_in_reference": head.feature_index.n_features,
        "n_probes_observed": scores.n_observed,
        "top3": top3,
        "per_model": per_model,
    }


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
    ap.add_argument("--router-checkpoint", type=Path, default=None)
    ap.add_argument("--router-threshold", type=float, default=0.9)
    ap.add_argument("--alma-reference-dir", type=Path, default=None)
    ap.add_argument("--alma-b0-checkpoint", type=Path, default=None)
    ap.add_argument("--alma-e1-checkpoint", type=Path, default=None)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    alma_args = (args.alma_reference_dir, args.alma_b0_checkpoint, args.alma_e1_checkpoint)
    two_stage = args.router_checkpoint is not None
    if two_stage != all(a is not None for a in alma_args):
        ap.error("--router-checkpoint requires all of --alma-reference-dir, "
                 "--alma-b0-checkpoint, --alma-e1-checkpoint (and vice versa)")

    cns = load_head(
        "cns", args.reference_dir, args.b0_checkpoint, args.e1_checkpoint, args.e1_kappa,
        feature_index_path=args.feature_index,
    )
    output: dict = {"sample_id": args.sample_id}

    if not two_stage:
        result = classify_with_head(
            cns, args.sample_counts, args.sample_id, args.top_n_loci, args.n_runner_up
        )
        output.update({k: v for k, v in result.items() if k != "head"})
        top = output["top3"][0]
        print(
            f"{args.sample_id}: {output['n_probes_observed']}/{output['n_probes_in_reference']} "
            f"probes observed -> top1={top['class_name']} (ensemble p={top['probability']:.3f})"
        )
    else:
        router, mask, checksum = load_router(args.router_checkpoint)
        if checksum != cns.feature_index.checksum:
            raise ValueError("router was trained on a different CNS feature index")
        m, u = load_counts_aligned(args.sample_counts, cns.feature_index.probe_ids)
        proba = route(router, mask, m, u)
        domain = decide_domain(proba, args.router_threshold)
        output["router"] = {
            "p_cns": float(proba[0]), "p_leukemia": float(proba[1]),
            "threshold": args.router_threshold, "domain": domain,
        }
        alma = load_head(
            "leukemia", args.alma_reference_dir, args.alma_b0_checkpoint,
            args.alma_e1_checkpoint, args.e1_kappa,
        )
        heads = {"cns": cns, "leukemia": alma}
        to_run = list(heads) if domain == UNDETERMINED else [domain]
        results = {
            name: classify_with_head(
                heads[name], args.sample_counts, args.sample_id,
                args.top_n_loci, args.n_runner_up,
            )
            for name in to_run
        }
        if domain == UNDETERMINED:
            output["heads"] = results  # both, for transparency; no diagnosis is asserted
            print(f"{args.sample_id}: domain undetermined (p_cns={proba[0]:.3f}, "
                  f"p_leukemia={proba[1]:.3f}) -> no head selected")
        else:
            output.update({k: v for k, v in results[domain].items()})
            top = output["top3"][0]
            print(f"{args.sample_id}: routed to {domain} (p={proba.max():.3f}) -> "
                  f"top1={top['class_name']} (ensemble p={top['probability']:.3f})")
        assert set(DOMAINS) == {"cns", "leukemia"}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
