#!/usr/bin/env python3
"""Evaluate the CNS-vs-leukemia router on cohorts whose domain is known.

Array cohorts: ALMA test (leukemia), GSE109379 (CNS; used for epoch selection).
Real ONT cohorts: Rapid-CNS2 and ONT-WGS and our own samples (CNS), and the 20
ALMA nanopore samples (leukemia; subtype labels are unavailable but the
domain is known). Reports argmax accuracy and, at ``--threshold``, how many
samples are routed vs abstained.

Usage:
    python scripts/evaluation/evaluate_router.py --output-dir outputs/results/router
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from neuromethyl_ont.data.alma import load_nanopore_betas  # noqa: E402
from neuromethyl_ont.data.array_reference import load_feature_index  # noqa: E402
from neuromethyl_ont.data.idat_reconstruction import load_mnp_preprocessed  # noqa: E402
from neuromethyl_ont.data.paths import get_data_root  # noqa: E402
from neuromethyl_ont.data.router_data import (  # noqa: E402
    binary_on_index,
    counts_to_binary,
)
from neuromethyl_ont.data.sequencing_simulator import binary_transform  # noqa: E402
from neuromethyl_ont.evaluation.cohort_scoring import extract_features  # noqa: E402
from neuromethyl_ont.training.router import DOMAINS, predict_domain_proba  # noqa: E402


def probe_list(path: Path) -> list[str]:
    return pd.read_csv(path, sep="\t")["probe_id"].astype(str).tolist()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path,
                    default=REPO_ROOT / "outputs/experiments/router_seed17/checkpoint_best.pt")
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    root = get_data_root()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    mask = np.asarray(ckpt["shared_probe_mask"], dtype=bool)
    model = nn.Linear(int(mask.size), len(DOMAINS))
    model.load_state_dict(ckpt["model_state_dict"])
    ref = root / "derived" / "reference" / "GSE90496"
    fi = load_feature_index(ref / "feature_index.json")
    if fi.checksum != ckpt["feature_index_checksum"]:
        raise RuntimeError("router was trained on a different feature index")
    mu = pd.read_parquet(ref / "probe_statistics.parquet").set_index("probe_id")["mean_beta"]
    mu = mu.reindex(fi.probe_ids).to_numpy()
    feature_pos = {p: i for i, p in enumerate(fi.probe_ids)}

    cohorts: dict[str, tuple[np.ndarray, str]] = {}
    ev = root / "derived" / "evaluation" / "ALMA" / "test"
    cohorts["ALMA test (array)"] = (
        binary_on_index(np.load(ev / "beta.npy", mmap_mode="r"),
                        probe_list(ev / "probe_ids.tsv"), fi, mask), "leukemia")
    nano = load_nanopore_betas()
    cohorts["ALMA nanopore (ONT)"] = (
        binary_on_index(nano.to_numpy(dtype=np.float64), nano.columns.astype(str).tolist(),
                        fi, mask), "leukemia")
    mnp = load_mnp_preprocessed(root / "datasets/GSE109379/processed/mnp_reconstruction")
    cohorts["GSE109379 (array)"] = (binary_on_index(mnp.beta, mnp.probe_ids, fi, mask), "cns")
    rapid = root / "derived" / "array_aligned" / "probe_level" / "RapidCNS2"
    for name, sub in (("Rapid-CNS2 (ONT, low cov)", "Rapid_CNS2"),
                      ("ONT-WGS (ONT, high cov)", "ONT_WGS")):
        files = sorted((rapid / sub).glob("*.parquet"))
        cohorts[name] = (counts_to_binary(extract_features(files, feature_pos, mu)["binary"],
                                          mask), "cns")
    ours = root / "derived" / "evaluation" / "our_samples_v1" / "counts" / "low_coverage"
    cohorts["Own MB/ETMR (ONT, ~10 min)"] = (
        counts_to_binary(binary_transform(np.load(ours / "m.npy"), np.load(ours / "u.npy")),
                         mask), "cns")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, per_sample = [], []
    for name, (x, truth) in cohorts.items():
        proba = predict_domain_proba(model, x, args.device)
        pred = proba.argmax(1)
        conf = proba.max(1)
        true_idx = DOMAINS.index(truth)
        routed = conf >= args.threshold
        rows.append({
            "cohort": name, "true_domain": truth, "n": len(x),
            "domain_accuracy": float((pred == true_idx).mean()),
            f"routed_at_{args.threshold}": float(routed.mean()),
            "routed_wrong": int(((pred != true_idx) & routed).sum()),
            "abstained": int((~routed).sum()),
            "min_p_true_domain": float(proba[:, true_idx].min()),
        })
        per_sample.append(pd.DataFrame({"cohort": name, "true_domain": truth,
                                        "p_cns": proba[:, 0], "p_leukemia": proba[:, 1]}))
    table = pd.DataFrame(rows)
    table.to_csv(args.output_dir / "router_metrics.tsv", sep="\t", index=False)
    pd.concat(per_sample).to_csv(args.output_dir / "router_predictions.tsv", sep="\t", index=False)
    print(table.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
