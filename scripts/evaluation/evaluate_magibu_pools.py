#!/usr/bin/env python3
"""Consolidated B0 vs C0 vs Sturgeon vs crossNN family-level comparison on
the data provider's pooled Magibu pseudo-samples. These pools carry no counts
(beta only), so E1 and the B0+E1 ensemble, which need sequencing counts,
cannot be scored here.

Scores B0 and C0 from the same beta.npy in one pass and adds Sturgeon and
crossNN's own predictions (scripts/competitors/run_sturgeon_predict.py /
scripts/competitors/run_crossnn_predict.py output), all onto the identical
family taxonomy, and writes one table with all models side by side per
pool -- the "high-coverage pools vs ~1x lowpass pool" comparison.

CAVEAT (read before quoting these numbers): every "sample" here is a pool of
5-125 real samples averaged into one pseudo-sample beta. This is a much
easier task than single-sample classification -- pooling itself cancels
noise that would otherwise make low-coverage single samples much harder to
classify. Treat these results as a feasibility demonstration of the
representation/architecture, not as an estimate of real single-sample
accuracy at any given coverage.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))


from neuromethyl_ont.data.sequencing_simulator import (  # noqa: E402
    binary_transform_full_array,
    continuous_transform_full_array,
)
from neuromethyl_ont.evaluation.classification import softmax_probs  # noqa: E402
from neuromethyl_ont.evaluation.family_taxonomy import (  # noqa: E402
    aggregate_to_family,
    family_metrics,
    sturgeon_model_class,
)
from neuromethyl_ont.models.evidence_linear import build_linear_classifier  # noqa: E402
from neuromethyl_ont.training.trainer import predict_logits  # noqa: E402


def load_checkpoint(path: Path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    classes = [
        label for label, _ in sorted(checkpoint["class_to_idx"].items(), key=lambda p: p[1])
    ]
    model = build_linear_classifier(checkpoint["n_features"], len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, classes


def reindex(probs: np.ndarray, source: list[str], target: list[str]) -> np.ndarray:
    """Move columns from a model's own family order onto the pool loop's
    reference family order (all family spaces are UNRESOLVED-complete, so
    this is a permutation, never a drop)."""
    out = np.zeros((len(probs), len(target)), dtype=np.float64)
    for col, family in enumerate(source):
        out[:, target.index(family)] = probs[:, col]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--magibu-root", type=Path, required=True, help="derived/evaluation/magibu_v1"
    )
    ap.add_argument("--b0-checkpoint", type=Path, required=True)
    ap.add_argument("--c0-checkpoint", type=Path, required=True)
    ap.add_argument("--taxonomy", type=Path, required=True)
    ap.add_argument("--reference-dir", type=Path, required=True, help="for C0's per-probe mean")
    ap.add_argument("--sturgeon-predictions", type=Path, required=True)
    ap.add_argument("--crossnn-predictions", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    taxonomy = pd.read_csv(args.taxonomy, sep="\t")
    family_by_class = {
        str(row.model_class): str(row.family_code).strip()
        for row in taxonomy.itertuples(index=False)
        if pd.notna(row.family_code) and str(row.family_code).strip()
    }

    b0_model, b0_classes = load_checkpoint(args.b0_checkpoint)
    c0_model, c0_classes = load_checkpoint(args.c0_checkpoint)
    if (
        set(b0_classes) != set(c0_classes)
        or set(b0_classes) != set(taxonomy.model_class)
    ):
        raise RuntimeError(
            "B0/C0 checkpoints and taxonomy must cover exactly the same classes"
        )
    mu = (
        pd.read_parquet(args.reference_dir / "probe_statistics.parquet")
        .set_index("probe_id")["mean_beta"]
    )

    def external_probs(
        path: Path, column_to_model_class=lambda c: c
    ) -> tuple[dict[str, np.ndarray], list[str]]:
        table = pd.read_csv(path, sep="\t")
        score_columns = [c for c in table.columns if c not in {"sample_id", "number_probes"}]
        model_classes = [column_to_model_class(c) for c in score_columns]
        probs, families = aggregate_to_family(
            table[score_columns].to_numpy(dtype=np.float64), model_classes, family_by_class
        )
        return {s: probs[i] for i, s in enumerate(table.sample_id)}, families

    # Sturgeon's own columns are its public label format ("Group - Code -
    # Subtype"), not our model_class strings -- translate before aggregating.
    # crossNN's columns are already model_class strings.
    sturgeon_by_sample, sturgeon_families = external_probs(
        args.sturgeon_predictions, sturgeon_model_class
    )
    crossnn_by_sample, crossnn_families = external_probs(args.crossnn_predictions)

    beta_root = args.magibu_root / "beta_only"
    manifests = sorted(beta_root.glob("*/manifest.tsv"))
    if not manifests:
        ap.error(f"no beta-only manifests under {beta_root}")

    result_rows: list[dict] = []
    sample_rows: list[pd.DataFrame] = []
    for manifest_path in manifests:
        pool = manifest_path.parent.name
        manifest = pd.read_csv(manifest_path, sep="\t")
        beta = np.load(manifest_path.parent / "beta.npy")
        if beta.shape[0] != len(manifest):
            raise RuntimeError(f"{pool}: beta rows and manifest disagree")

        b0_probs = softmax_probs(predict_logits(b0_model, binary_transform_full_array(beta)))
        b0_fam, families = aggregate_to_family(b0_probs, b0_classes, family_by_class)
        probe_ids = pd.read_csv(manifest_path.parent / "probe_ids.tsv", sep="\t")["probe_id"]
        mu_i = mu.reindex(probe_ids).to_numpy()
        c0_probs = softmax_probs(
            predict_logits(c0_model, continuous_transform_full_array(beta, mu_i))
        )
        c0_fam, c0_families = aggregate_to_family(c0_probs, c0_classes, family_by_class)
        if c0_families != families:
            raise RuntimeError(f"{pool}: B0/C0 family layouts disagree")

        if not set(manifest.sample_id).issubset(sturgeon_by_sample):
            raise RuntimeError(f"{pool}: some samples lack a Sturgeon prediction")
        if not set(manifest.sample_id).issubset(crossnn_by_sample):
            raise RuntimeError(f"{pool}: some samples lack a crossNN prediction")
        sturgeon_stack = np.stack([sturgeon_by_sample[s] for s in manifest.sample_id])
        sturgeon_fam = reindex(sturgeon_stack, sturgeon_families, families)
        crossnn_stack = np.stack([crossnn_by_sample[s] for s in manifest.sample_id])
        crossnn_fam = reindex(crossnn_stack, crossnn_families, families)

        truth = manifest.family_code.to_numpy()
        model_probs = {
            "B0": b0_fam, "C0": c0_fam,
            "Sturgeon": sturgeon_fam, "crossNN": crossnn_fam,
        }
        for model_name, probs in model_probs.items():
            result_rows.append(
                {"pool": pool, "model": model_name, **family_metrics(truth, probs, families)}
            )

        keep_cols = ["sample_id", "benchmark", "pool_size", "ground_truth_raw", "family_code"]
        frame = manifest[keep_cols].copy()
        for model_name, probs in model_probs.items():
            frame[f"{model_name}_predicted_family"] = [families[i] for i in probs.argmax(axis=1)]
            frame[f"{model_name}_true_family_probability"] = probs[
                np.arange(len(frame)), [families.index(f) for f in truth]
            ]
        sample_rows.append(frame)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(result_rows)
    summary.to_csv(
        args.output_dir / "magibu_pool_model_comparison_metrics.tsv", sep="\t", index=False
    )
    pd.concat(sample_rows, ignore_index=True).to_csv(
        args.output_dir / "magibu_pool_model_comparison_samples.tsv", sep="\t", index=False
    )
    pivot = summary.pivot(index="pool", columns="model", values="family_accuracy")
    print(pivot.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
