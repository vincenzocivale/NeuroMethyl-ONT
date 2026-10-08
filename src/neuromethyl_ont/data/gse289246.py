"""Parser for GSE289246 (ONT bedMethyl -> canonical coordinate-level rows).

GSE289246 ships one gzip bedMethyl-style file per sample
(``datasets/GSE289246/raw/extracted/<GSM>_*.bed.gz``), genome-wide, one row
per covered CpG position, with a fixed 12-column header:

    chrom start end item score strand thickStart thickEnd itemRgb
    coverage percentMeth Probe_ID_450K

This is NOT the standard modkit-pileup bedMethyl layout (18 columns, no
header) -- ``Probe_ID_450K`` is a 450K-probe annotation GEO added to a
subset of rows. Only those probe-annotated rows (~2% of a file) carry
information this project's array-aligned probe-level representation uses;
unannotated genome-wide rows are dropped before any aggregation, matching
"array_aligned/probe_level" scope (docs/DATA_LAYOUT.md).

``coverage``/``percentMeth`` are per-genomic-position counts, never
per-probe -- ``methylated_count``/``unmethylated_count`` are derived here
(rounded) and then summed across duplicate coordinates per probe by
:func:`neuromethyl_ont.data.canonical.aggregate_to_probe_level`, exactly as
for Rapid-CNS2/ONT-WGS (see docs/RAPIDCNS2_CANONICAL_PREPROCESSING.md).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from neuromethyl_ont.data.schemas import CG_PROBE_PATTERN

BEDMETHYL_COLUMNS = [
    "chrom",
    "start",
    "end",
    "item",
    "score",
    "strand",
    "thickStart",
    "thickEnd",
    "itemRgb",
    "coverage",
    "percentMeth",
    "Probe_ID_450K",
]

# Columns actually needed downstream; the rest (item/score/strand/thick*/itemRgb)
# are read-but-discarded genomic-track bookkeeping this project doesn't use.
USED_COLUMNS = ["chrom", "start", "end", "coverage", "percentMeth", "Probe_ID_450K"]


class GSE289246ParseError(ValueError):
    """Raised when a GSE289246 bedMethyl file doesn't match the expected schema."""


def read_gse289246_sample(path: str | Path) -> pd.DataFrame:
    """Read one GSE289246 ``*.CG.bed.gz`` / ``*.CpG.bed.gz`` file into
    coordinate-level rows, keeping only probe-annotated CpGs.

    Returns columns ``chrom, start, end, probe_id, coverage,
    methylation_fraction, methylated_count, unmethylated_count`` --
    :func:`neuromethyl_ont.data.canonical.to_coordinate_level` still needs to
    attach ``sample_id`` before aggregation.

    Raises :class:`GSE289246ParseError` on a header/column mismatch, a
    ``Probe_ID_450K`` value that isn't a cg-number, or coverage/percentMeth
    outside their valid range -- this project's policy (see
    ``docs/RAPIDCNS2_CANONICAL_PREPROCESSING.md``) is to fail loudly on an
    unrecognized schema rather than guess.
    """
    path = Path(path)
    if not path.exists():
        raise GSE289246ParseError(f"not found: {path}")

    header = pd.read_csv(path, sep="\t", nrows=0).columns.tolist()
    missing = set(BEDMETHYL_COLUMNS) - set(header)
    if missing:
        raise GSE289246ParseError(f"{path}: missing expected column(s) {sorted(missing)}")

    df = pd.read_csv(
        path,
        sep="\t",
        usecols=USED_COLUMNS,
        dtype={
            "chrom": "string",
            "start": "int64",
            "end": "int64",
            "coverage": "int64",
            "percentMeth": "float64",
        },
    )
    df = df.rename(columns={"Probe_ID_450K": "probe_id"})
    df["probe_id"] = df["probe_id"].astype("string")
    df = df[df["probe_id"].notna() & (df["probe_id"] != "")].reset_index(drop=True)
    # GEO's annotation column also contains Infinium SNP controls (rs IDs).
    # They are not CpG probes and therefore cannot be represented in the
    # 450K-aligned classifier/Sturgeon feature space.
    df = df[~df["probe_id"].str.fullmatch(r"rs\d+")].reset_index(drop=True)

    if df.empty:
        return pd.DataFrame(
            columns=[
                "chrom", "start", "end", "probe_id", "coverage",
                "methylation_fraction", "methylated_count", "unmethylated_count",
            ]
        )

    bad_probe_id = ~df["probe_id"].str.match(CG_PROBE_PATTERN)
    if bad_probe_id.any():
        example = df.loc[bad_probe_id, "probe_id"].iloc[0]
        raise GSE289246ParseError(
            f"{path}: {int(bad_probe_id.sum())} row(s) with a Probe_ID_450K value that "
            f"isn't a cg-number, e.g. {example!r}"
        )

    bad_coverage = df["coverage"] <= 0
    if bad_coverage.any():
        raise GSE289246ParseError(
            f"{path}: {int(bad_coverage.sum())} probe-annotated row(s) with coverage <= 0"
        )
    bad_pct = (df["percentMeth"] < 0) | (df["percentMeth"] > 100)
    if bad_pct.any():
        raise GSE289246ParseError(
            f"{path}: {int(bad_pct.sum())} probe-annotated row(s) with percentMeth outside "
            f"[0, 100]"
        )

    coverage = df["coverage"].to_numpy()
    methylated_count = np.rint(coverage * (df["percentMeth"].to_numpy() / 100.0)).astype("int64")
    methylated_count = np.clip(methylated_count, 0, coverage)
    unmethylated_count = coverage - methylated_count

    out = pd.DataFrame(
        {
            "chrom": df["chrom"],
            "start": df["start"].astype("Int64"),
            "end": df["end"].astype("Int64"),
            "probe_id": df["probe_id"],
            "coverage": pd.array(coverage, dtype="Int64"),
            "methylated_count": pd.array(methylated_count, dtype="Int64"),
            "unmethylated_count": pd.array(unmethylated_count, dtype="Int64"),
        }
    )
    out["methylation_fraction"] = (
        out["methylated_count"].astype("Float64") / out["coverage"].astype("Float64")
    ).astype("float64")
    return out
