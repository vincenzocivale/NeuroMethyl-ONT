#!/usr/bin/env python3
"""B0 vs C0 vs EB1 vs Sturgeon vs crossNN on Rapid-CNS2 / ONT-WGS: this
project's primary *real* (not synthetic, not pooled) coverage-aware
benchmark pair (docs/RAPIDCNS2_CANONICAL_PREPROCESSING.md, EVIDENCE_LINEAR_V1.md
#16). Rapid_CNS2 (n=300, median mean coverage ~2.0x) is the low-coverage,
intraoperative-like cohort; ONT_WGS (n=39, median mean coverage ~29.6x) is
the high-coverage, end-of-run-like cohort -- real per-sample varying
coverage, not a pooled average and not a synthetic Poisson draw, unlike
the Magibu pools (compare_magibu_pools.py) or the GSE90496 holdout curve
(docs/EVIDENCE_BAYES_V1.md #7). This is the one place EB1's coverage-aware
claim can actually be tested end to end.

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
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from neuromethyl_ont.data.array_reference import load_feature_index  # noqa: E402
from neuromethyl_ont.data.rapidcns2_labels import load_rapidcns2_labels  # noqa: E402
from neuromethyl_ont.data.sequencing_simulator import evidence_transform  # noqa: E402
from neuromethyl_ont.evaluation.classification import softmax_probs  # noqa: E402
from neuromethyl_ont.evaluation.family_taxonomy import (  # noqa: E402
    aggregate_to_family,
    family_metrics,
    sturgeon_model_class,
)
from neuromethyl_ont.models.evidence_bayes import predict_log_joint_torch  # noqa: E402
from neuromethyl_ont.models.evidence_linear import build_linear_classifier  # noqa: E402
from neuromethyl_ont.training.trainer import predict_logits  # noqa: E402

BINARY_THRESHOLD = 0.6


def load_torch_checkpoint(path: Path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    classes = [
        label for label, _ in sorted(checkpoint["class_to_idx"].items(), key=lambda p: p[1])
    ]
    model = build_linear_classifier(checkpoint["n_features"], len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, classes


class EvidenceBayesCheckpoint:
    """Just enough of EvidenceBayesModel's interface for predict_log_joint_torch."""

    def __init__(self, checkpoint_dir: Path):
        npz = np.load(checkpoint_dir / "checkpoint.npz")
        meta = json.loads((checkpoint_dir / "checkpoint_meta.json").read_text())
        self.log_prior = npz["log_prior"]
        self.alpha = npz["alpha"]
        self.beta = npz["beta"]
        self.weight = npz["weight"]
        self.kappa = npz["kappa"]
        self.classes = meta["classes"]
        self.probe_ids = meta["probe_ids"]

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    @property
    def n_probes(self) -> int:
        return len(self.probe_ids)


def extract_features(
    files: list[Path], feature_pos: dict[str, int], mu: np.ndarray
) -> dict[str, np.ndarray]:
    """One parquet read per sample -> binary/continuous/count features plus
    per-sample mean coverage, all sharing the same feature_index column
    order so every model scores the exact same evidence."""
    n = len(feature_pos)
    binary = np.zeros((len(files), n), dtype=np.float32)
    continuous = np.zeros((len(files), n), dtype=np.float32)
    m = np.zeros((len(files), n), dtype=np.float64)
    u = np.zeros((len(files), n), dtype=np.float64)
    mean_coverage = np.zeros(len(files), dtype=np.float64)
    for row, path in enumerate(files):
        frame = pd.read_parquet(
            path,
            columns=["probe_id", "coverage", "methylation_fraction", "methylated_count",
                     "unmethylated_count"],
        )
        mean_coverage[row] = frame["coverage"].mean()
        pos = frame["probe_id"].map(feature_pos)
        valid = pos.notna() & (frame["coverage"] > 0) & frame["methylation_fraction"].notna()
        idx = pos.loc[valid].astype(np.int64).to_numpy()
        beta = frame.loc[valid, "methylation_fraction"].to_numpy()
        binary[row, idx] = np.where(beta > BINARY_THRESHOLD, 1.0, -1.0)
        continuous[row, idx] = beta - mu[idx]
        m[row, idx] = frame.loc[valid, "methylated_count"].to_numpy()
        u[row, idx] = frame.loc[valid, "unmethylated_count"].to_numpy()
    return {
        "sample_id": [path.stem for path in files],
        "binary": binary,
        "continuous": continuous,
        "m": m,
        "u": u,
        "mean_coverage": mean_coverage,
    }


def reindex(probs: np.ndarray, source: list[str], target: list[str]) -> np.ndarray:
    out = np.zeros((len(probs), len(target)), dtype=np.float64)
    for col, family in enumerate(source):
        out[:, target.index(family)] = probs[:, col]
    return out


def score_cohort(
    files: list[Path],
    feature_pos: dict[str, int],
    mu: np.ndarray,
    b0_model,
    b0_classes: list[str],
    c0_model,
    c0_classes: list[str],
    eb1: EvidenceBayesCheckpoint,
    family_by_class: dict[str, str],
    device: str,
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
        raise RuntimeError("B0/C0 family layouts disagree")
    eb1_log_joint = predict_log_joint_torch(eb1, features["m"], features["u"], device=device)
    eb1_probs = softmax_probs(eb1_log_joint)
    eb1_fam, eb1_families = aggregate_to_family(eb1_probs, eb1.classes, family_by_class)
    eb1_fam = reindex(eb1_fam, eb1_families, families)

    model_probs = {"B0": b0_fam, "C0": c0_fam, "EB1": eb1_fam}

    if e1_model is not None:
        evidence = evidence_transform(features["m"], features["u"], mu, e1_kappa).astype(
            np.float32
        )
        e1_probs = softmax_probs(predict_logits(e1_model, evidence))
        e1_fam, e1_families = aggregate_to_family(e1_probs, e1_classes, family_by_class)
        e1_fam = reindex(e1_fam, e1_families, families)
        model_probs["E1"] = e1_fam
        model_probs["B0+E1 (avg)"] = 0.5 * b0_fam + 0.5 * e1_fam

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
    ap.add_argument("--b0-checkpoint", type=Path, required=True)
    ap.add_argument("--c0-checkpoint", type=Path, required=True)
    ap.add_argument("--eb1-checkpoint-dir", type=Path, required=True)
    ap.add_argument(
        "--e1-checkpoint", type=Path, default=None, help="optional; omit to skip E1 entirely"
    )
    ap.add_argument("--e1-kappa", type=float, default=2.0)
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
    ap.add_argument("--device", default="cuda")
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

    b0_model, b0_classes = load_torch_checkpoint(args.b0_checkpoint)
    c0_model, c0_classes = load_torch_checkpoint(args.c0_checkpoint)
    eb1 = EvidenceBayesCheckpoint(args.eb1_checkpoint_dir)
    if set(b0_classes) != set(c0_classes) or set(b0_classes) != set(eb1.classes):
        raise RuntimeError("B0/C0/EB1 must cover exactly the same 91 classes")

    e1_model, e1_classes = (None, None)
    if args.e1_checkpoint is not None:
        e1_model, e1_classes = load_torch_checkpoint(args.e1_checkpoint)
        if set(e1_classes) != set(b0_classes):
            raise RuntimeError("E1 must cover exactly the same 91 classes as B0/C0/EB1")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    rapid_files = sorted(args.rapid_dir.glob("*.parquet"))
    ont_files = sorted(args.ont_dir.glob("*.parquet"))
    print(f"Rapid_CNS2: {len(rapid_files)} files; ONT_WGS: {len(ont_files)} files", flush=True)

    rapid_frame, rapid_probs, families = score_cohort(
        rapid_files, feature_pos, mu, b0_model, b0_classes, c0_model, c0_classes, eb1,
        family_by_class, args.device, e1_model, e1_classes, args.e1_kappa,
    )
    ont_frame, ont_probs, ont_families = score_cohort(
        ont_files, feature_pos, mu, b0_model, b0_classes, c0_model, c0_classes, eb1,
        family_by_class, args.device, e1_model, e1_classes, args.e1_kappa,
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
    model_names = ["B0", "C0", "EB1", "Sturgeon", "crossNN"]
    if e1_model is not None:
        model_names[3:3] = ["E1", "B0+E1 (avg)"]

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
