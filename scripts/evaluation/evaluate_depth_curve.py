#!/usr/bin/env python3
"""Fixed synthetic depth-curve evaluation across B0/C0/E1 (docs/EVIDENCE_LINEAR_V1.md #13/#15).

Loads one checkpoint per representation (as written by
scripts/training/train_classifier.py), builds ONE fixed set of synthetic
sequencing realizations on the validation cohort, and evaluates every
checkpoint on the same realizations at every depth -- so the resulting
``depth_curve.tsv`` compares B0/C0/E1 on identical synthetic evidence.

Usage:
    python scripts/evaluation/evaluate_depth_curve.py \\
        --reference-dir "$NEUROMETHYL_DATA_ROOT/derived/reference/GSE90496" \\
        --validation-beta PATH --validation-probe-ids PATH --validation-labels PATH \\
        --checkpoint B0=outputs/experiments/b0_run1/checkpoint_best.pt \\
        --checkpoint C0=outputs/experiments/c0_run1/checkpoint_best.pt \\
        --checkpoint E1=outputs/experiments/e1_run1/checkpoint_best.pt \\
        --output outputs/experiments/v1_representation_ablation/depth_curve.tsv
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

from neuromethyl_ont.data.array_reference import (  # noqa: E402
    align_to_feature_index,
    load_feature_index,
)
from neuromethyl_ont.evaluation.depth_curve import run_depth_curve  # noqa: E402
from neuromethyl_ont.models.evidence_linear import build_linear_classifier  # noqa: E402


def load_probe_ids(path: Path) -> list[str]:
    if path.suffix == ".npy":
        return [str(p) for p in np.load(path, allow_pickle=True)]
    return pd.read_csv(path, sep="\t")["probe_id"].astype(str).tolist()


def parse_checkpoint_arg(value: str) -> tuple[str, Path]:
    name, _, path = value.partition("=")
    if not path:
        raise argparse.ArgumentTypeError(f"expected NAME=PATH, got {value!r}")
    return name, Path(path)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--reference-dir", type=Path, required=True)
    ap.add_argument("--validation-beta", type=Path, required=True)
    ap.add_argument("--validation-probe-ids", type=Path, required=True)
    ap.add_argument("--validation-labels", type=Path, required=True)
    ap.add_argument(
        "--checkpoint",
        action="append",
        type=parse_checkpoint_arg,
        required=True,
        dest="checkpoints",
        help="NAME=PATH, repeatable (e.g. --checkpoint B0=... --checkpoint C0=... "
        "--checkpoint E1=...)",
    )
    ap.add_argument(
        "--kappa", type=float, default=2.0, help="kappa used to build E1 features at eval time"
    )
    ap.add_argument(
        "--seed", type=int, default=0, help="fixed realization seed, shared across all checkpoints"
    )
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    feature_index = load_feature_index(args.reference_dir / "feature_index.json")
    probe_stats = pd.read_parquet(args.reference_dir / "probe_statistics.parquet")
    mu = probe_stats.set_index("probe_id").loc[feature_index.probe_ids, "mean_beta"].to_numpy()

    val_probe_ids = load_probe_ids(args.validation_probe_ids)
    val_beta_raw = np.load(args.validation_beta).astype(np.float64)
    val_beta = align_to_feature_index(val_beta_raw, val_probe_ids, feature_index)
    val_labels_df = pd.read_csv(args.validation_labels, sep="\t")

    models: dict[str, torch.nn.Module] = {}
    representations: dict[str, str] = {}
    class_to_idx = None
    for name, path in args.checkpoints:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if class_to_idx is None:
            class_to_idx = checkpoint["class_to_idx"]
        elif checkpoint["class_to_idx"] != class_to_idx:
            print(
                f"WARNING: checkpoint {name} has a different class mapping than earlier "
                "checkpoints",
                file=sys.stderr,
            )
        model = build_linear_classifier(
            n_features=checkpoint["n_features"], n_classes=len(checkpoint["class_to_idx"])
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        models[name] = model
        representations[name] = checkpoint["representation"]

    keep = val_labels_df["label"].isin(class_to_idx)
    if not keep.all():
        print(
            f"WARNING: dropping {(~keep).sum()} validation sample(s) with unseen labels",
            file=sys.stderr,
        )
    val_beta = val_beta[keep.to_numpy()]
    y_val = np.array(
        [class_to_idx[label] for label in val_labels_df.loc[keep, "label"]], dtype=np.int64
    )

    depth_curve = run_depth_curve(
        models=models, representations=representations, beta=val_beta, y_true=y_val,
        mu=mu, kappa=args.kappa, seed=args.seed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    depth_curve.to_csv(args.output, sep="\t", index=False)
    print(f"Wrote {len(depth_curve)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
