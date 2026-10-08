#!/usr/bin/env python3
"""Train the CNS-vs-leukemia domain router (src/neuromethyl_ont/training/router.py).

Train: GSE90496 (CNS) vs ALMA train split (leukemia).
Validation (epoch selection): GSE109379 (CNS) vs ALMA val split (leukemia).
Saves outputs/experiments/router_seed17/{checkpoint_best.pt,training_history.tsv}.
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

from neuromethyl_ont.data.array_reference import load_feature_index  # noqa: E402
from neuromethyl_ont.data.idat_reconstruction import load_mnp_preprocessed  # noqa: E402
from neuromethyl_ont.data.paths import get_data_root  # noqa: E402
from neuromethyl_ont.data.router_data import binary_on_index, shared_probe_mask  # noqa: E402
from neuromethyl_ont.training.router import (  # noqa: E402
    DOMAINS,
    predict_domain_proba,
    train_router,
)


def probe_list(path: Path) -> list[str]:
    return pd.read_csv(path, sep="\t")["probe_id"].astype(str).tolist()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", default="router_seed17")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--keep-min", type=float, default=0.2,
                    help="lowest fraction of probes kept by the training-time masking")
    args = ap.parse_args()

    root = get_data_root()
    cns_ref = root / "derived" / "reference" / "GSE90496"
    alma_ref = root / "derived" / "reference" / "ALMA_discovery_train"
    alma_eval = root / "derived" / "evaluation" / "ALMA"
    fi = load_feature_index(cns_ref / "feature_index.json")
    alma_probes = probe_list(alma_ref / "probe_ids.tsv")
    mask = shared_probe_mask(fi, alma_probes)
    print(f"router features: {int(mask.sum())} shared probes of {fi.n_features}", flush=True)

    def build(beta_path: Path, probes: list[str]) -> np.ndarray:
        return binary_on_index(np.load(beta_path, mmap_mode="r"), probes, fi, mask)

    x_cns = build(cns_ref / "beta.npy", probe_list(cns_ref / "probe_ids.tsv"))
    x_leu = build(alma_ref / "beta.npy", alma_probes)
    x_leu_val = build(alma_eval / "val" / "beta.npy", probe_list(alma_eval / "val" / "probe_ids.tsv"))
    mnp = load_mnp_preprocessed(
        root / "datasets" / "GSE109379" / "processed" / "mnp_reconstruction"
    )
    x_cns_val = binary_on_index(mnp.beta, mnp.probe_ids, fi, mask)
    print(f"train: cns={len(x_cns)} leukemia={len(x_leu)}; "
          f"val: cns={len(x_cns_val)} leukemia={len(x_leu_val)}", flush=True)

    x_train = np.concatenate([x_cns, x_leu])
    y_train = np.concatenate([np.zeros(len(x_cns)), np.ones(len(x_leu))])
    x_val = np.concatenate([x_cns_val, x_leu_val])
    y_val = np.concatenate([np.zeros(len(x_cns_val)), np.ones(len(x_leu_val))])
    del x_cns, x_leu

    result = train_router(x_train, y_train, x_val, y_val, device=args.device, seed=args.seed,
                          keep_range=(args.keep_min, 1.0))
    proba = predict_domain_proba(result.model, x_val, args.device)
    print(f"best epoch {result.best_epoch}; val accuracy "
          f"{(proba.argmax(1) == y_val).mean():.4f}", flush=True)

    out = REPO_ROOT / "outputs" / "experiments" / args.run_id
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": result.model.state_dict(), "domains": DOMAINS,
                "shared_probe_mask": mask, "best_epoch": result.best_epoch,
                "feature_index_checksum": fi.checksum}, out / "checkpoint_best.pt")
    pd.DataFrame(result.history).to_csv(out / "training_history.tsv", sep="\t", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
