#!/usr/bin/env python3
"""Score the ALMA leukemia heads (Binary, Beta, Evidence and Consensus) on the held-out splits.

ALMA cohorts are arrays (beta only, no counts), so Evidence is evaluated in its
full-array limit, as during training validation. The test split should be
read once, after model selection on val is finished.

Usage:
    python scripts/evaluation/evaluate_alma.py --output-dir outputs/results/alma
"""

from __future__ import annotations

import argparse
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
from neuromethyl_ont.data.paths import get_data_root  # noqa: E402
from neuromethyl_ont.data.sequencing_simulator import encode_full_array  # noqa: E402
from neuromethyl_ont.evaluation.classification import (  # noqa: E402
    depth_curve_metrics,
    softmax_probs,
)
from neuromethyl_ont.evaluation.cohort_scoring import load_torch_checkpoint  # noqa: E402
from neuromethyl_ont.training.trainer import predict_logits  # noqa: E402


def load_probe_ids(path: Path) -> list[str]:
    return pd.read_csv(path, sep="\t")["probe_id"].astype(str).tolist()


REPRESENTATION = {"Binary": "binary", "Beta": "continuous", "Evidence": "evidence"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiments-dir", type=Path, default=REPO_ROOT / "outputs/experiments")
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    root = get_data_root()
    ref_dir = root / "derived" / "reference" / "ALMA_discovery_train"
    feature_index = load_feature_index(ref_dir / "feature_index.json")
    mu = pd.read_parquet(ref_dir / "probe_statistics.parquet").set_index("probe_id")["mean_beta"]
    mu = mu.reindex(feature_index.probe_ids).to_numpy()

    models = {}
    for name in REPRESENTATION:
        model, classes = load_torch_checkpoint(
            args.experiments_dir / f"alma_{name.lower()}_seed17" / "checkpoint_best.pt"
        )
        models[name] = model
    class_to_idx = {c: i for i, c in enumerate(classes)}
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows, per_class = [], []
    for split in args.splits:
        d = root / "derived" / "evaluation" / "ALMA" / split
        beta = align_to_feature_index(
            np.load(d / "beta.npy").astype(np.float64), load_probe_ids(d / "probe_ids.tsv"),
            feature_index,
        )
        labels = pd.read_csv(d / "labels.tsv", sep="\t")
        y = labels["label"].map(class_to_idx).to_numpy()
        if np.isnan(y.astype(float)).any():
            raise RuntimeError(f"{split}: labels outside the training classes")
        y = y.astype(np.int64)

        probs = {}
        for name, model in models.items():
            x = encode_full_array(REPRESENTATION[name], beta, mu=mu, threshold=0.6)
            probs[name] = softmax_probs(predict_logits(model, x))
        probs["Consensus"] = 0.5 * (probs["Binary"] + probs["Evidence"])

        for name, p in probs.items():
            metrics = depth_curve_metrics(np.log(np.clip(p, 1e-12, None)), y)
            rows.append({"split": split, "model": name, "n_samples": len(y), **metrics})
            pred = p.argmax(axis=1)
            for cls, idx in class_to_idx.items():
                m = y == idx
                if m.any():
                    per_class.append(
                        {"split": split, "model": name, "class": cls, "n": int(m.sum()),
                         "recall": float((pred[m] == idx).mean())}
                    )
        if split == "test":
            out = labels.assign(**{f"{n}_pred": [classes[i] for i in p.argmax(1)]
                                   for n, p in probs.items()})
            out.to_csv(args.output_dir / "alma_test_predictions.tsv", sep="\t", index=False)

    table = pd.DataFrame(rows)
    table.to_csv(args.output_dir / "alma_metrics.tsv", sep="\t", index=False)
    pd.DataFrame(per_class).to_csv(args.output_dir / "alma_per_class.tsv", sep="\t", index=False)
    cols = ["split", "model", "n_samples", "accuracy", "balanced_accuracy", "macro_f1",
            "top3_accuracy", "nll", "ece"]
    print(table[[c for c in cols if c in table.columns]].round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
