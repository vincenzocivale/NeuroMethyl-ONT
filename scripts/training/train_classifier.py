#!/usr/bin/env python3
"""Train one of B0/C0/E1 (docs/EVIDENCE_LINEAR_V1.md) and write run outputs.

    GSE90496 reference (build_array_reference.py output)
        -> LinearClassifier, representation from --model-config
        -> depth-augmented training (trainer.train_linear_classifier)
        -> full-array GSE109379 validation
        -> outputs/experiments/<run_id>/

Same script for all three representations -- only ``--model-config`` changes
(configs/models/{binary_linear,continuous_linear,evidence_linear_v1}.yaml).
The fixed synthetic depth curve (spec #13/#15) is a separate step:
scripts/evaluation/evaluate_depth_curve.py, run against this script's
checkpoint.

Usage:
    python scripts/training/train_classifier.py \\
        --model-config configs/models/evidence_linear_v1.yaml \\
        --reference-dir "$NEUROMETHYL_DATA_ROOT/derived/reference/GSE90496" \\
        --validation-beta PATH --validation-probe-ids PATH --validation-labels PATH \\
        --run-id e1_run1
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import classification_report

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from neuromethyl_ont.data.array_reference import (  # noqa: E402
    align_to_feature_index,
    load_array_reference,
    load_feature_index,
)
from neuromethyl_ont.data.sequencing_simulator import encode_full_array  # noqa: E402
from neuromethyl_ont.evaluation.classification import depth_curve_metrics  # noqa: E402
from neuromethyl_ont.models.evidence_linear import build_linear_classifier  # noqa: E402
from neuromethyl_ont.training.trainer import TrainingConfig, train_linear_classifier  # noqa: E402


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def load_probe_ids(path: Path) -> list[str]:
    if path.suffix == ".npy":
        return [str(p) for p in np.load(path, allow_pickle=True)]
    return pd.read_csv(path, sep="\t")["probe_id"].astype(str).tolist()


def build_training_config(model_config: dict, representation: str) -> TrainingConfig:
    optimizer = model_config.get("optimizer", {})
    training = model_config.get("training", {})
    loss = model_config.get("loss", {})
    model = model_config.get("model", {})
    return TrainingConfig(
        representation=representation,
        lr=float(optimizer.get("lr", 3.0e-4)),
        weight_decay=float(optimizer.get("weight_decay", 1.0e-4)),
        batch_size=int(training.get("batch_size", 16)),
        max_epochs=int(training.get("max_epochs", 80)),
        early_stopping_patience=int(training.get("early_stopping_patience", 12)),
        mixed_precision=str(training.get("mixed_precision", "bf16")),
        kappa=float(model.get("kappa", 2.0)),
        binary_threshold=float(model.get("binary_threshold", 0.6)),
        seed=int(model_config.get("seed", 17)),
        loss_name=str(loss.get("name", "cross_entropy")),
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model-config", type=Path, required=True)
    ap.add_argument(
        "--reference-dir", type=Path, required=True, help="build_array_reference.py output dir"
    )
    ap.add_argument(
        "--validation-beta", type=Path, required=True, help="samples x probes .npy, GSE109379 order"
    )
    ap.add_argument("--validation-probe-ids", type=Path, required=True)
    ap.add_argument(
        "--validation-labels", type=Path, required=True, help=".tsv with sample_id, label"
    )
    ap.add_argument("--validation-dataset", default="GSE109379")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    model_config = yaml.safe_load(args.model_config.read_text())
    representation = model_config["model"]["representation"]

    ref = load_array_reference(args.reference_dir)
    feature_index = load_feature_index(args.reference_dir / "feature_index.json")

    classes = sorted(set(ref.labels))
    if len(classes) != 91:
        print(f"WARNING: expected 91 training classes, found {len(classes)}", file=sys.stderr)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_train = np.array([class_to_idx[label] for label in ref.labels], dtype=np.int64)

    val_probe_ids = load_probe_ids(args.validation_probe_ids)
    val_beta_raw = np.load(args.validation_beta).astype(np.float64)
    val_beta = align_to_feature_index(val_beta_raw, val_probe_ids, feature_index)
    val_labels_df = pd.read_csv(args.validation_labels, sep="\t")
    unknown = sorted(set(val_labels_df["label"]) - set(class_to_idx))
    if unknown:
        print(
            f"WARNING: {len(unknown)} validation class(es) unseen in training: {unknown}",
            file=sys.stderr,
        )
    keep = val_labels_df["label"].isin(class_to_idx)
    val_beta = val_beta[keep.to_numpy()]
    y_val = np.array(
        [class_to_idx[label] for label in val_labels_df.loc[keep, "label"]], dtype=np.int64
    )

    model = build_linear_classifier(n_features=feature_index.n_features, n_classes=len(classes))
    training_config = build_training_config(model_config, representation)
    training_config.device = args.device

    result = train_linear_classifier(
        model=model,
        beta_train=ref.beta,
        y_train=y_train,
        beta_val=val_beta,
        y_val=y_val,
        mu=ref.probe_mean,
        config=training_config,
    )
    if result.best_state_dict is not None:
        model.load_state_dict(result.best_state_dict)

    model.eval()
    with torch.no_grad():
        x_val_full = encode_full_array(
            representation, val_beta, mu=ref.probe_mean, threshold=training_config.binary_threshold
        )
        val_logits = model(torch.as_tensor(x_val_full, dtype=torch.float32)).numpy()
    val_metrics = depth_curve_metrics(val_logits, y_val)
    val_prediction = val_logits.argmax(axis=1)

    output_dir = args.output_dir or (REPO_ROOT / "outputs" / "experiments" / args.run_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "config.yaml").write_text(yaml.safe_dump(model_config, sort_keys=False))

    history_rows = [
        {
            "epoch": r.epoch,
            "train_loss": r.train_loss,
            "val_loss": r.val_loss,
            "val_accuracy": r.val_accuracy,
        }
        for r in result.history
    ]
    pd.DataFrame(history_rows).to_csv(output_dir / "training_history.tsv", sep="\t", index=False)

    (output_dir / "validation_metrics.json").write_text(json.dumps(val_metrics, indent=2))

    report = classification_report(
        y_val, val_prediction, labels=list(range(len(classes))), target_names=classes,
        output_dict=True, zero_division=0,
    )
    pd.DataFrame(report).T.rename_axis("class_name").reset_index().to_csv(
        output_dir / "per_class_metrics.tsv", sep="\t", index=False
    )

    torch.save(
        {
            "model_state_dict": result.best_state_dict,
            "class_to_idx": class_to_idx,
            "representation": representation,
            "n_features": feature_index.n_features,
        },
        output_dir / "checkpoint_best.pt",
    )

    metadata = {
        "git_commit": git_commit(),
        "feature_index_checksum": feature_index.checksum,
        "training_dataset": feature_index.source_dataset,
        "validation_dataset": args.validation_dataset,
        "n_features": feature_index.n_features,
        "n_classes": len(classes),
        "class_mapping_version": 1,
        "representation": representation,
        "best_epoch": result.best_epoch,
        "best_val_accuracy": result.best_val_accuracy,
        "stopped_early": result.stopped_early,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"Best epoch {result.best_epoch}: val_accuracy={result.best_val_accuracy:.4f}")
    print(f"Wrote run outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
