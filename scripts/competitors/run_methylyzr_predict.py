#!/usr/bin/env python3
"""Run MethyLYZR's pre-trained Naive Bayes model
(`third_party/MethyLYZR/model/{betas_mean,W_RELIEF}.feather` +
`class_priors.csv`, downloaded separately per its README -- not tracked in
this repo) on per-read fingerprint feathers produced by
`third_party/MethyLYZR/scripts/bam2feather.py`.

Unlike Sturgeon/crossNN (which take a probe/beta table), MethyLYZR scores a
per-read fingerprint -- one row per (read, CpG) pair with the read's own
methylation call, not a per-probe aggregate -- so its input must come from
a raw BAM with MM/ML tags via bam2feather.py, never from the beta-only
Magibu pools or the coordinate-level Rapid_CNS2/GSE289246 data this
project's other competitor runners consume.

Requires MethyLYZR's own virtualenv (`third_party/MethyLYZR/venv`, built
per its README from `requirements.txt`) -- run this script with that
venv's Python, not this repo's own `.venv`.

Output matches the shape of run_crossnn_predict.py/run_sturgeon_predict.py:
one row per sample, `sample_id`, `number_probes` (distinct EPIC probes
observed in the feather, pre-filtering), then one column per MethyLYZR
class with its posterior probability.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "third_party" / "MethyLYZR"))

from MethyLYZR import predict_sample  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--feather-dir", type=Path, required=True,
        help="one .feather per sample (bam2feather.py output)",
    )
    ap.add_argument(
        "--centroids", type=Path,
        default=REPO_ROOT / "third_party/MethyLYZR/model/betas_mean.feather",
    )
    ap.add_argument(
        "--weights", type=Path,
        default=REPO_ROOT / "third_party/MethyLYZR/model/W_RELIEF.feather",
    )
    ap.add_argument(
        "--priors", type=Path,
        default=REPO_ROOT / "third_party/MethyLYZR/model/class_priors.csv",
    )
    ap.add_argument("--min-noise", type=float, default=0.05)
    ap.add_argument("--meth-lower-bound", type=float, default=0.8)
    ap.add_argument("--meth-upper-bound", type=float, default=0.2)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    for path in (args.centroids, args.weights, args.priors):
        if not path.exists():
            ap.error(
                f"missing {path} -- download from the link in third_party/MethyLYZR/README.md "
                "(Data Requirements) and place under third_party/MethyLYZR/model/"
            )

    betas_mean = pd.read_feather(args.centroids).set_index("index")
    w_relief = pd.read_feather(args.weights).set_index("index")
    class_frequency = pd.read_csv(args.priors, index_col=0).squeeze("columns")

    feathers = sorted(args.feather_dir.glob("*.feather"))
    if not feathers:
        ap.error(f"no feather inputs in {args.feather_dir}")

    rows: list[dict] = []
    for number, feather_path in enumerate(feathers, start=1):
        sample_id = feather_path.stem
        posterior = predict_sample(
            sample_id=sample_id,
            sample_dir=feather_path,
            centroids=betas_mean,
            W=w_relief,
            class_frequency=class_frequency,
            min_noise=args.min_noise,
            methylation_lower_bound=args.meth_lower_bound,
            methylation_upper_bound=args.meth_upper_bound,
        )
        n_probes = pd.read_feather(feather_path, columns=["epic_id"])["epic_id"].nunique()
        row = {"sample_id": sample_id, "number_probes": int(n_probes)}
        row.update(posterior.to_dict())
        rows.append(row)
        print(f"progress: {number}/{len(feathers)} ({sample_id})", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, sep="\t", index=False)
    print(f"Wrote {len(rows)} prediction(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
