#!/usr/bin/env python3
"""Export a GSM -> IDAT-basename manifest from a dataset's ``*_family.soft.gz``,
for the R IDAT importer (``scripts/data/idat_to_beta_mnp.R``).

Usage:
    python scripts/data/export_idat_sample_manifest.py \\
        "$NEUROMETHYL_DATA_ROOT/datasets/GSE109379" [--soft-family PATH] [--output PATH]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from neuromethyl_ont.data.geo_series import GeoSeriesParseError, parse_soft_family  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("dataset_dir", type=Path)
    ap.add_argument("--soft-family", type=Path, default=None)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    soft_family = args.soft_family
    if soft_family is None:
        matches = list((dataset_dir / "metadata").glob("*_family.soft.gz"))
        if len(matches) != 1:
            print(
                f"expected exactly one *_family.soft.gz under {dataset_dir}/metadata, "
                f"found {matches}",
                file=sys.stderr,
            )
            return 2
        soft_family = matches[0]

    labels = parse_soft_family(soft_family)

    missing = labels["idat_basename"].isna()
    if missing.any():
        print(
            f"WARNING: {int(missing.sum())}/{len(labels)} sample(s) have no IDAT "
            "supplementary files and will be dropped from the manifest",
            file=sys.stderr,
        )
    labels = labels.loc[~missing].reset_index(drop=True)

    output = args.output or (dataset_dir / "processed" / "idat_sample_manifest.tsv")
    output.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(output, sep="\t", index=False)
    print(f"Wrote {len(labels)} sample(s) to {output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GeoSeriesParseError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
