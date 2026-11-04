#!/usr/bin/env python3
"""Binary vs Beta vs Evidence vs Consensus vs Sturgeon vs crossNN on Rapid-CNS2 / ONT-WGS: this
project's primary *real* (not synthetic, not pooled) coverage-aware
benchmark pair (docs/RAPIDCNS2_CANONICAL_PREPROCESSING.md, EVIDENCE_LINEAR_V1.md
#16). Rapid_CNS2 (n=300, median mean coverage ~2.0x) is the low-coverage,
intraoperative-like cohort; ONT_WGS (n=39, median mean coverage ~29.6x) is
the high-coverage, end-of-run-like cohort -- real per-sample varying
coverage, not a pooled average and not a synthetic Poisson draw, unlike
the Magibu pools (evaluate_magibu_pools.py) or the GSE90496 holdout curve.

Rapid_CNS2 has ground truth (Supplementary Tables 1/2); ONT_WGS does not,
so it is scored only on predicted-family agreement across models, not
accuracy -- family-level metrics are reported for Rapid_CNS2 only.
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from neuromethyl_ont.data.array_reference import load_feature_index  # noqa: E402
from neuromethyl_ont.data.rapidcns2_labels import load_rapidcns2_labels  # noqa: E402
from neuromethyl_ont.data.sequencing_simulator import evidence_transform  # noqa: E402
from neuromethyl_ont.evaluation.classification import softmax_probs  # noqa: E402
from neuromethyl_ont.evaluation.cohort_scoring import (  # noqa: E402
    ensemble_average,
    extract_features,
    load_torch_checkpoint,
    reindex,
)
from neuromethyl_ont.evaluation.family_taxonomy import (  # noqa: E402
    aggregate_to_family,
    family_metrics,
    sturgeon_model_class,
)
from neuromethyl_ont.training.trainer import predict_logits  # noqa: E402


def score_cohort(
    files: list[Path],
    feature_pos: dict[str, int],
    mu: np.ndarray,
    b0_model,
    b0_classes: list[str],
    c0_model,
    c0_classes: list[str],
    family_by_class: dict[str, str],
    e1_model=None,
    e1_classes: list[str] | None = None,
    e1_kappa: float = 2.0,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], list[str]]:
    features = extract_features(files, feature_pos, mu)
    b0_probs = softmax_probs(predict_logits(b0_model, features["binary"]))
    b0_fam, families = aggregate_to_family(b0_probs, b0_classes, family_by_class)
    c0_probs = softmax_probs(predict_logits(c0_model, features["continuous"]))
    c0_fam, c0_families = aggregate_to_family(c0_probs, c0_classes, family_by_class)
    if c0_families != families:
        raise RuntimeError("Binary/Beta family layouts disagree")
    model_probs = {"Binary": b0_fam, "Beta": c0_fam}

    if e1_model is not None:
        evidence = evidence_transform(features["m"], features["u"], mu, e1_kappa).astype(
            np.float32
        )
        e1_probs = softmax_probs(predict_logits(e1_model, evidence))
        e1_fam, e1_families = aggregate_to_family(e1_probs, e1_classes, family_by_class)
        e1_fam = reindex(e1_fam, e1_families, families)
        model_probs["Evidence"] = e1_fam
        model_probs["Consensus"] = ensemble_average(b0_fam, e1_fam)

    frame = pd.DataFrame(
        {"sample_id": features["sample_id"], "mean_coverage": features["mean_coverage"]}
    )
    for name, probs in model_probs.items():
        frame[f"{name}_predicted_family"] = [families[i] for i in probs.argmax(axis=1)]
        frame[f"{name}_confidence"] = probs.max(axis=1)
    return frame, model_probs, families


def add_external_model(
    frame: pd.DataFrame,
    model_probs: dict[str, np.ndarray],
    families: list[str],
    name: str,
    predictions_path: Path,
    family_by_class: dict[str, str],
    column_to_model_class,
) -> None:
    table = pd.read_csv(predictions_path, sep="\t")
    score_columns = [c for c in table.columns if c not in {"sample_id", "number_probes"}]
    model_classes = [column_to_model_class(c) for c in score_columns]
    probs, ext_families = aggregate_to_family(
        table[score_columns].to_numpy(dtype=np.float64), model_classes, family_by_class
    )
    probs = reindex(probs, ext_families, families)
    by_sample = {s: probs[i] for i, s in enumerate(table.sample_id)}
    if not set(frame.sample_id).issubset(by_sample):
        raise RuntimeError(
            f"{name}: some samples in this cohort lack a prediction in {predictions_path}"
        )
    stacked = np.stack([by_sample[s] for s in frame.sample_id])
    model_probs[name] = stacked
    frame[f"{name}_predicted_family"] = [families[i] for i in stacked.argmax(axis=1)]
    frame[f"{name}_confidence"] = stacked.max(axis=1)


def per_class_accuracy(
    scored: pd.DataFrame,
    rows: np.ndarray,
    model_probs: dict[str, np.ndarray],
    families: list[str],
    model_names: list[str],
    min_n: int = 5,
) -> pd.DataFrame:
    """Per-true-family top-1 accuracy (recall), families with >= ``min_n``
    labelled samples only -- feeds ``plot_b0_per_class_heatmaps.py``/
    ``plot_results.py``'s per-class panels."""
    family_idx = {f: i for i, f in enumerate(families)}
    out_rows = []
    for family, mask in scored.groupby("family_code").groups.items():
        n = len(mask)
        if n < min_n:
            continue
        local_rows = scored.index.get_indexer(mask)
        row = {"family": family, "n": n}
        for name in model_names:
            preds = model_probs[name][rows[local_rows]].argmax(axis=1)
            row[name] = float(np.mean(preds == family_idx[family]))
        out_rows.append(row)
    return pd.DataFrame(out_rows).sort_values("n", ascending=False).reset_index(drop=True)


def pairwise_agreement(frame: pd.DataFrame, model_names: list[str]) -> pd.DataFrame:
    rows = []
    for a, b in combinations(model_names, 2):
        agree = (frame[f"{a}_predicted_family"] == frame[f"{b}_predicted_family"]).mean()
        rows.append({"model_a": a, "model_b": b, "family_top1_agreement": agree})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--reference-dir", type=Path, required=True)
    ap.add_argument("--binary-checkpoint", "--b0-checkpoint", type=Path, required=True)
    ap.add_argument("--beta-checkpoint", "--c0-checkpoint", type=Path, required=True)
    ap.add_argument(
        "--evidence-checkpoint", "--e1-checkpoint", type=Path, default=None, help="optional; omit to skip Evidence entirely"
    )
    ap.add_argument("--evidence-kappa", "--e1-kappa", type=float, default=2.0)
    ap.add_argument("--taxonomy", type=Path, required=True)
    ap.add_argument("--rapid-dir", type=Path, required=True)
    ap.add_argument("--rapid-labels-xlsx", type=Path, required=True)
    ap.add_argument("--sturgeon-rapid", type=Path, required=True)
    ap.add_argument("--crossnn-rapid", type=Path, required=True)
    ap.add_argument("--ont-dir", type=Path, required=True)
    ap.add_argument("--sturgeon-ont", type=Path, required=True)
    ap.add_argument("--crossnn-ont", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument(
        "--perclass-output", type=Path,
        default=REPO_ROOT / "outputs" / "benchmarks" / "rapidcns2_perclass.tsv",
        help="per-family (n>=5) top-1 accuracy table -- plot_results.py's default input",
    )
    args = ap.parse_args()

    feature_index = load_feature_index(args.reference_dir / "feature_index.json")
    mu = (
        pd.read_parquet(args.reference_dir / "probe_statistics.parquet")
        .set_index("probe_id")["mean_beta"]
        .reindex(feature_index.probe_ids)
        .to_numpy()
    )
    feature_pos = {probe: i for i, probe in enumerate(feature_index.probe_ids)}

    taxonomy = pd.read_csv(args.taxonomy, sep="\t")
    family_by_class = {
        str(row.model_class): str(row.family_code).strip()
        for row in taxonomy.itertuples(index=False)
        if pd.notna(row.family_code) and str(row.family_code).strip()
    }

    b0_model, b0_classes = load_torch_checkpoint(args.binary_checkpoint)
    c0_model, c0_classes = load_torch_checkpoint(args.beta_checkpoint)
    if set(b0_classes) != set(c0_classes):
        raise RuntimeError("Binary/Beta must cover exactly the same classes")

    e1_model, e1_classes = (None, None)
    if args.evidence_checkpoint is not None:
        e1_model, e1_classes = load_torch_checkpoint(args.evidence_checkpoint)
        if set(e1_classes) != set(b0_classes):
            raise RuntimeError("Evidence must cover exactly the same classes as Binary/Beta")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    rapid_files = sorted(args.rapid_dir.glob("*.parquet"))
    ont_files = sorted(args.ont_dir.glob("*.parquet"))
    print(f"Rapid_CNS2: {len(rapid_files)} files; ONT_WGS: {len(ont_files)} files", flush=True)

    rapid_frame, rapid_probs, families = score_cohort(
        rapid_files, feature_pos, mu, b0_model, b0_classes, c0_model, c0_classes,
        family_by_class, e1_model, e1_classes, args.evidence_kappa,
    )
    ont_frame, ont_probs, ont_families = score_cohort(
        ont_files, feature_pos, mu, b0_model, b0_classes, c0_model, c0_classes,
        family_by_class, e1_model, e1_classes, args.evidence_kappa,
    )
    if ont_families != families:
        raise RuntimeError("Rapid_CNS2/ONT_WGS family layouts disagree")

    add_external_model(
        rapid_frame, rapid_probs, families, "Sturgeon", args.sturgeon_rapid,
        family_by_class, sturgeon_model_class,
    )
    add_external_model(
        rapid_frame, rapid_probs, families, "crossNN", args.crossnn_rapid,
        family_by_class, lambda c: c,
    )
    add_external_model(
        ont_frame, ont_probs, families, "Sturgeon", args.sturgeon_ont,
        family_by_class, sturgeon_model_class,
    )
    add_external_model(
        ont_frame, ont_probs, families, "crossNN", args.crossnn_ont,
        family_by_class, lambda c: c,
    )
    model_names = ["Binary", "Beta", "Sturgeon", "crossNN"]
    if e1_model is not None:
        model_names[2:2] = ["Evidence", "Consensus"]

    labels = load_rapidcns2_labels(args.rapid_labels_xlsx, set(b0_classes)).labels
    truth_family = labels.assign(family_code=labels.label.map(family_by_class))
    scored = rapid_frame.merge(truth_family, on="sample_id", how="inner")
    scored = scored.loc[scored.family_code.notna()].copy()
    row_pos = {sample: i for i, sample in enumerate(rapid_frame.sample_id)}
    rows = np.array([row_pos[s] for s in scored.sample_id], dtype=np.int64)

    metric_rows = []
    for name in model_names:
        metric_rows.append(
            {
                "cohort": "Rapid_CNS2 (low coverage, median ~2x)",
                "model": name,
                "n_samples": len(scored),
                **family_metrics(scored.family_code.to_numpy(), rapid_probs[name][rows], families),
            }
        )
    metrics = pd.DataFrame(metric_rows)

    rapid_frame.to_csv(
        args.output_dir / "rapidcns2_low_coverage_predictions.tsv", sep="\t", index=False
    )
    ont_frame.to_csv(
        args.output_dir / "ont_wgs_high_coverage_predictions.tsv", sep="\t", index=False
    )
    metrics.to_csv(
        args.output_dir / "rapidcns2_model_comparison_metrics.tsv", sep="\t", index=False
    )
    perclass = per_class_accuracy(scored, rows, rapid_probs, families, model_names)
    args.perclass_output.parent.mkdir(parents=True, exist_ok=True)
    perclass.to_csv(args.perclass_output, sep="\t", index=False)
    ont_agreement = pairwise_agreement(ont_frame, model_names)
    ont_agreement.to_csv(
        args.output_dir / "ont_wgs_pairwise_agreement.tsv", sep="\t", index=False
    )

    summary = {
        "policy": "Rapid_CNS2 scored on ground truth (family_metrics); ONT_WGS is unlabelled, "
        "scored only on cross-model predicted-family agreement.",
        "n_excluded_unresolved_truth": int(len(labels) - len(scored)),
        "rapid_cns2_mean_coverage": {
            "min": float(rapid_frame.mean_coverage.min()),
            "median": float(rapid_frame.mean_coverage.median()),
            "max": float(rapid_frame.mean_coverage.max()),
        },
        "ont_wgs_mean_coverage": {
            "min": float(ont_frame.mean_coverage.min()),
            "median": float(ont_frame.mean_coverage.median()),
            "max": float(ont_frame.mean_coverage.max()),
        },
    }
    (args.output_dir / "comparison_metadata.json").write_text(json.dumps(summary, indent=2))

    display_cols = ["model", "family_accuracy", "balanced_accuracy", "macro_f1", "nll"]
    print("\n=== Rapid_CNS2 (low coverage, labelled) ===")
    print(metrics[display_cols].to_string(index=False))
    print("\n=== ONT_WGS (high coverage, unlabelled) pairwise top-1 family agreement ===")
    print(ont_agreement.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
