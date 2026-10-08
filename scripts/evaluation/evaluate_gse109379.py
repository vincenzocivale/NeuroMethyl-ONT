#!/usr/bin/env python3
"""B0 vs C0 on GSE109379 -- this project's own registry
(registry/datasets.tsv, docs/EXPERIMENT_POLICY.md) names this the
validation/model-selection cohort, but no model had ever been scored on it
(a previously open gap). Closes it.

GSE109379's own precomputed matrix (`GSE109379_processed_data.txt.gz`) is
truncated upstream (docs/GSE109379_IDAT_RECONSTRUCTION.md), so this script
reads the IDAT-reconstructed beta matrix written by
`scripts/data/idat_to_beta_mnp.R` instead -- real per-sample array beta, no
coverage/counts (450K array), same shape as the Magibu beta-only pools
(evaluate_magibu_pools.py) but *individual* real samples, not pooled
averages, and with real full-91-class ground truth (not just family), from
GSE109379's own SOFT metadata (`methylation_class` column already extracted
into the IDAT sample manifest by export_idat_sample_manifest.py).

This cohort is beta-only (no counts), so E1 and the B0+E1 ensemble, which need
sequencing counts, cannot be scored here.

Sturgeon/crossNN are scored too, if their prediction tables are passed
(--sturgeon-predictions/--crossnn-predictions, built by
build_sturgeon_bed_gse109379.py + run_sturgeon_predict.py/
run_crossnn_predict.py) -- family-level only, same as every other cohort in
this project, since both competitors use their own class vocabulary and
only agree with this project's taxonomy at the family level
(sturgeon_model_class translates Sturgeon's public labels first). Omit both
flags to score B0/C0 only.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from neuromethyl_ont.data.array_reference import (  # noqa: E402
    align_to_feature_index,
    load_feature_index,
)
from neuromethyl_ont.data.idat_reconstruction import load_mnp_preprocessed  # noqa: E402
from neuromethyl_ont.data.sequencing_simulator import (  # noqa: E402
    binary_transform_full_array,
    continuous_transform_full_array,
)
from neuromethyl_ont.evaluation.classification import (  # noqa: E402
    expected_calibration_error,
    negative_log_likelihood,
    reliability_bins,
    softmax_probs,
    top_k_accuracy,
    top_k_table,
)
from neuromethyl_ont.evaluation.cohort_scoring import load_torch_checkpoint  # noqa: E402
from neuromethyl_ont.evaluation.family_taxonomy import (  # noqa: E402
    UNRESOLVED,
    aggregate_to_family,
    family_metrics,
    sturgeon_model_class,
)
from neuromethyl_ont.evaluation.metrics import classification_metrics  # noqa: E402
from neuromethyl_ont.training.trainer import predict_logits  # noqa: E402


def _normalize_class(label: str) -> str:
    """Collapse repeated whitespace -- GSE109379's SOFT metadata has at least
    one class ("PIN T,  PB B", double space) that otherwise silently fails
    to match the training checkpoint's class_to_idx (checked: every other
    class matches exactly, this is the only offender)."""
    return re.sub(r"\s+", " ", label).strip()


def external_probs(
    path: Path, family_by_class: dict[str, str], column_to_model_class=lambda c: c
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Sturgeon/crossNN's own predictions table -> {sample_id: family_probs}, families.

    Same pattern as evaluate_rapidcns2.py/evaluate_gse289246.py: each
    competitor's columns are its own class vocabulary, translated to this
    project's model_class strings (Sturgeon needs sturgeon_model_class first;
    crossNN's columns already are model_class strings) before aggregating.
    """
    table = pd.read_csv(path, sep="\t")
    score_columns = [c for c in table.columns if c not in {"sample_id", "number_probes"}]
    model_classes = [column_to_model_class(c) for c in score_columns]
    probs, families = aggregate_to_family(
        table[score_columns].to_numpy(dtype=np.float64), model_classes, family_by_class
    )
    return {s: probs[i] for i, s in enumerate(table.sample_id)}, families


def reindex(probs: np.ndarray, source: list[str], target: list[str]) -> np.ndarray:
    out = np.zeros((len(probs), len(target)), dtype=np.float64)
    for col, family in enumerate(source):
        out[:, target.index(family)] = probs[:, col]
    return out


def class_level_metrics(probs: np.ndarray, y_true: np.ndarray) -> dict[str, float]:
    prediction = probs.argmax(axis=1)
    metrics = dict(classification_metrics(y_true, prediction))
    metrics["top3_accuracy"] = top_k_accuracy(probs, y_true, k=3)
    metrics["nll"] = negative_log_likelihood(probs, y_true)
    metrics["ece"] = expected_calibration_error(probs, y_true)
    return metrics


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--mnp-dir", type=Path, required=True,
        help="output dir of scripts/data/idat_to_beta_mnp.R (meth/unmeth/beta/detection_p .f8)",
    )
    ap.add_argument(
        "--manifest", type=Path, required=True,
        help="datasets/GSE109379/processed/idat_sample_manifest.tsv (gsm, methylation_class, ...)",
    )
    ap.add_argument("--reference-dir", type=Path, required=True, help="derived/reference/GSE90496")
    ap.add_argument("--b0-checkpoint", type=Path, required=True)
    ap.add_argument("--c0-checkpoint", type=Path, required=True)
    ap.add_argument("--taxonomy", type=Path, required=True)
    ap.add_argument("--sturgeon-predictions", type=Path, default=None)
    ap.add_argument("--crossnn-predictions", type=Path, default=None)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    feature_index = load_feature_index(args.reference_dir / "feature_index.json")
    mu = (
        pd.read_parquet(args.reference_dir / "probe_statistics.parquet")
        .set_index("probe_id")["mean_beta"]
        .reindex(feature_index.probe_ids)
        .to_numpy()
    )

    taxonomy = pd.read_csv(args.taxonomy, sep="\t")
    family_by_class = {
        str(row.model_class): str(row.family_code).strip()
        for row in taxonomy.itertuples(index=False)
        if pd.notna(row.family_code) and str(row.family_code).strip()
    }

    b0_model, b0_classes = load_torch_checkpoint(args.b0_checkpoint)
    c0_model, c0_classes = load_torch_checkpoint(args.c0_checkpoint)
    if set(b0_classes) != set(c0_classes):
        raise RuntimeError("B0/C0 must cover exactly the same classes")
    class_to_idx = {c: i for i, c in enumerate(b0_classes)}
    # The checkpoint's own class strings are not internally whitespace-clean
    # (e.g. "PIN T,  PB B" has a double space) -- normalize both sides
    # identically rather than assume the manifest is the only offender.
    normalized_to_checkpoint_class = {_normalize_class(c): c for c in b0_classes}
    if len(normalized_to_checkpoint_class) != len(b0_classes):
        raise RuntimeError(
            "whitespace-normalizing checkpoint classes collides two distinct classes"
        )

    print(f"Loading MNP-reconstructed beta from {args.mnp_dir} ...", flush=True)
    mnp = load_mnp_preprocessed(args.mnp_dir)
    beta = align_to_feature_index(mnp.beta, mnp.probe_ids, feature_index)
    print(f"{len(mnp.sample_ids)} samples x {beta.shape[1]} feature-index probes "
          f"(reconstructed at {len(mnp.probe_ids)} probes)", flush=True)

    manifest = pd.read_csv(args.manifest, sep="\t")
    label_by_gsm = {
        row.gsm: normalized_to_checkpoint_class.get(_normalize_class(row.methylation_class))
        for row in manifest.itertuples(index=False)
    }
    unmapped = sorted(
        {row.methylation_class for row in manifest.itertuples(index=False)
         if row.gsm in mnp.sample_ids and label_by_gsm[row.gsm] is None}
    )
    if unmapped:
        raise RuntimeError(f"GSE109379 class(es) not in the 91-class checkpoint: {unmapped}")

    sample_ids, y_true = [], []
    beta_rows = []
    n_no_label = 0
    for row, gsm in enumerate(mnp.sample_ids):
        label = label_by_gsm.get(gsm)
        if label is None:
            n_no_label += 1
            continue
        sample_ids.append(gsm)
        y_true.append(class_to_idx[label])
        beta_rows.append(beta[row])
    if n_no_label:
        print(f"WARNING: {n_no_label} reconstructed sample(s) had no manifest label, skipped",
              file=sys.stderr)
    beta = np.stack(beta_rows)
    y_true = np.array(y_true, dtype=np.int64)
    print(f"Scoring {len(sample_ids)} labelled samples", flush=True)

    b0_probs = softmax_probs(predict_logits(b0_model, binary_transform_full_array(beta)))
    c0_probs = softmax_probs(
        predict_logits(c0_model, continuous_transform_full_array(beta, mu))
    )
    model_probs = {"B0": b0_probs, "C0": c0_probs}

    # A handful of classes (e.g. MELCYT) have no established family assignment
    # in the taxonomy (docs note "no supported family assignment") -- same
    # UNRESOLVED bucket aggregate_to_family already uses for predicted probs,
    # applied here to true labels too rather than crashing on a KeyError.
    true_families = np.array(
        [family_by_class.get(b0_classes[i], UNRESOLVED) for i in y_true], dtype=object
    )

    class_rows, family_rows = [], []
    frame = pd.DataFrame({
        "sample_id": sample_ids,
        "true_class": [b0_classes[i] for i in y_true],
        "true_family": true_families,
    })
    for name, probs in model_probs.items():
        class_rows.append(
            {"model": name, "n_samples": len(sample_ids), **class_level_metrics(probs, y_true)}
        )
        fam_probs, families = aggregate_to_family(probs, b0_classes, family_by_class)
        family_rows.append({"model": name, **family_metrics(true_families, fam_probs, families)})
        frame[f"{name}_predicted_class"] = [b0_classes[i] for i in probs.argmax(axis=1)]
        frame[f"{name}_true_class_probability"] = probs[np.arange(len(y_true)), y_true]
        frame[f"{name}_predicted_family"] = [families[i] for i in fam_probs.argmax(axis=1)]

    # families/true_families are fixed by the loop above (B0/C0 share one
    # family layout, checked via aggregate_to_family's own class coverage) --
    # reuse the same `families` order for Sturgeon/crossNN.
    external = [
        ("Sturgeon", args.sturgeon_predictions, sturgeon_model_class),
        ("crossNN", args.crossnn_predictions, lambda c: c),
    ]
    for name, path, column_to_model_class in external:
        if path is None:
            continue
        by_sample, ext_families = external_probs(path, family_by_class, column_to_model_class)
        missing = set(sample_ids) - set(by_sample)
        if missing:
            raise RuntimeError(f"{name}: {len(missing)} GSE109379 sample(s) lack a prediction "
                                f"in {path} (e.g. {sorted(missing)[:3]})")
        stacked = reindex(np.stack([by_sample[s] for s in sample_ids]), ext_families, families)
        family_rows.append({"model": name, **family_metrics(true_families, stacked, families)})
        frame[f"{name}_predicted_family"] = [families[i] for i in stacked.argmax(axis=1)]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "gse109379_predictions.tsv", sep="\t", index=False)
    pd.DataFrame(class_rows).to_csv(
        args.output_dir / "gse109379_class_metrics.tsv", sep="\t", index=False
    )
    pd.DataFrame(family_rows).to_csv(
        args.output_dir / "gse109379_family_metrics.tsv", sep="\t", index=False
    )

    # Per-model calibration (reliability diagram) and per-sample top-3 detail
    # -- the class-level metrics table above already has the scalar top3_accuracy/
    # ece, this is the underlying per-bin/per-sample data needed to actually
    # draw a reliability diagram or list which samples top-3 recovers.
    for name, probs in model_probs.items():
        bins = reliability_bins(probs, y_true, n_bins=10)
        pd.DataFrame(bins).to_csv(
            args.output_dir / f"gse109379_{name.lower()}_calibration.tsv", sep="\t", index=False
        )
        top3 = top_k_table(probs, y_true, b0_classes, k=3)
        top3_frame = pd.DataFrame(top3)
        top3_frame.insert(0, "sample_id", sample_ids)
        top3_frame.to_csv(
            args.output_dir / f"gse109379_{name.lower()}_top3.tsv", sep="\t", index=False
        )

    print("\n=== GSE109379, 91-class accuracy ===")
    print(pd.DataFrame(class_rows)[["model", "accuracy", "top3_accuracy", "balanced_accuracy",
                                     "macro_f1", "nll", "ece"]].to_string(index=False))
    print("\n=== GSE109379, family-level accuracy ===")
    print(pd.DataFrame(family_rows)[["model", "family_accuracy", "balanced_accuracy",
                                      "macro_f1", "nll"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
