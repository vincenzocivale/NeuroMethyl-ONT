#!/usr/bin/env python3
"""Run Sturgeon (github.com/marcpaga/sturgeon, installed system-wide for
python3.9 -- Sturgeon requires python <3.10, this project's own .venv is
3.10+, so this script is standalone and must be run with that separate
interpreter) over a directory of bed files built by
build_sturgeon_bed_inputs.py, and write one combined predictions TSV.

This calls sturgeon.prediction.{load_model,predict_sample} directly rather
than the ``sturgeon predict`` CLI -- functionally identical (same functions),
just without going through argparse/per-file logging setup, and loading the
model once instead of once per sample.

Usage (note: python3.9, not this repo's .venv):
    /usr/bin/python3.9 scripts/benchmarks/run_sturgeon_predict.py \\
        --bed-dir outputs/benchmarks/sturgeon/rapidcns2_bed \\
        --model-file ~/sturgeon/model/general.zip \\
        --output outputs/benchmarks/sturgeon/rapidcns2_predictions.tsv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

try:
    from sturgeon.prediction import load_model, predict_sample
except ImportError as exc:  # pragma: no cover - environment guard, not test-covered
    sys.exit(
        "Could not import sturgeon. This script must be run with the python3.9 "
        "interpreter that has `sturgeon` installed (not this repo's .venv), e.g.:\n"
        "  /usr/bin/python3.9 scripts/benchmarks/run_sturgeon_predict.py ...\n"
        f"(original error: {exc})"
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--bed-dir", type=Path, required=True)
    ap.add_argument("--model-file", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    bed_files = sorted(args.bed_dir.glob("*.bed"))
    if not bed_files:
        print(f"no *.bed files found under {args.bed_dir}", file=sys.stderr)
        return 2

    print(f"Loading model: {args.model_file}")
    inference_session, probes_df, decoding_dict, temperatures, merge_dict = load_model(
        str(args.model_file)
    )

    rows = []
    for i, bed_file in enumerate(bed_files, start=1):
        sample_id = bed_file.stem
        prediction_df = predict_sample(
            inference_session=inference_session,
            bed_file=str(bed_file),
            decoding_dict=decoding_dict,
            probes_df=probes_df,
            temperatures=temperatures,
            merge_dict=merge_dict,
        )
        prediction_df.insert(0, "sample_id", sample_id)
        rows.append(prediction_df)
        if i % 25 == 0 or i == len(bed_files):
            print(f"progress: {i}/{len(bed_files)}")

    combined = pd.concat(rows, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output, sep="\t", index=False)
    print(f"Wrote {len(combined)} prediction(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
