#!/usr/bin/env python3
"""Run the official crossNN Capper v11b4 checkpoint on binary probe BEDs.

The input is a five-column bed (chrom/chromStart/chromEnd/methylation_call/
probe_id): observed probes have a binary call (0/1) and absent probes
remain unobserved. crossNN's documented ``NN_classifier.predict`` interface
takes exactly ``probe_id`` plus calls in ``{-1, +1}``, so calls are mapped
0 -> -1 and 1 -> +1; no coverage is created.

Point ``--bed-dir`` at bed files built with
``build_sturgeon_bed_*.py --caller crossnn`` (crossNN's own single-0.6-
cutoff decision rule, ``crossnn_adapter.probe_level_to_crossnn_bed``), not
at Sturgeon's own ``--caller sturgeon`` (0.7/0.3 dead zone) output -- the
two are not interchangeable: Sturgeon's dead zone silently withholds every
probe in (0.3, 0.7) that crossNN's own pipeline would still call, which is
exactly the discrepancy ``--caller`` was added to fix (see
``crossnn_adapter.py``'s docstring). GSE209865 is the one exception: its
bed comes straight from GEO-published binary calls with no threshold step,
so the same bed is correct for both.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "third_party" / "crossNN"))

from NN_model import NN_classifier  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bed-dir", type=Path, required=True)
    ap.add_argument(
        "--model",
        type=Path,
        default=REPO_ROOT / "third_party" / "crossNN" / "models" / "Capper_et_al_NN.pkl",
    )
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    beds = sorted(path for path in args.bed_dir.glob("*.bed") if path.name != "input_manifest.tsv")
    if not beds:
        ap.error(f"no BED inputs in {args.bed_dir}")
    model = NN_classifier(args.model)
    rows = []
    for number, bed_path in enumerate(beds, start=1):
        bed = pd.read_csv(
            bed_path,
            sep="\t",
            names=["chrom", "start", "end", "methylation_call", "probe_id"],
            header=0,
        )
        calls = bed[["probe_id", "methylation_call"]].copy()
        calls["methylation_call"] = calls["methylation_call"].map({0: -1.0, 1: 1.0})
        scores, labels, n_features = model.predict(calls)
        row = {"sample_id": bed_path.stem, "number_probes": n_features}
        row.update(dict(zip(labels, scores, strict=True)))
        rows.append(row)
        if number % 25 == 0 or number == len(beds):
            print(f"progress: {number}/{len(beds)}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, sep="\t", index=False)
    print(f"Wrote {len(rows)} prediction(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
