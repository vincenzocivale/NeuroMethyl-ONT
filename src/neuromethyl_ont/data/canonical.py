"""Conversion between the two canonical methylation representations.

- ``to_coordinate_level``: attach sample_id and enforce the coordinate-level
  column contract. One row per genomic observation; no aggregation.
- ``aggregate_to_probe_level``: evidence-weighted aggregation over
  ``probe_id`` within a sample. Read counts are summed, never averaged as
  betas -- this is required to preserve total coverage and to correctly
  weight probes annotated at more than one coordinate (see project brief on
  MNP-Flex bedtools-intersect duplication).
"""

from __future__ import annotations

import pandas as pd

from neuromethyl_ont.data.schemas import COORDINATE_COLUMNS, PROBE_COLUMNS


def to_coordinate_level(df: pd.DataFrame, sample_id: str) -> pd.DataFrame:
    """Attach ``sample_id`` and return columns in canonical coordinate-level order.

    `df` is expected to already have the columns produced by
    :func:`neuromethyl_ont.data.rapidcns2.read_rapidcns2_sample`.
    """
    missing = set(COORDINATE_COLUMNS) - {"sample_id"} - set(df.columns)
    if missing:
        raise ValueError(f"to_coordinate_level: input is missing columns {sorted(missing)}")

    out = df.copy()
    out.insert(0, "sample_id", pd.array([sample_id] * len(out), dtype="string"))
    return out[COORDINATE_COLUMNS].reset_index(drop=True)


def aggregate_to_probe_level(coord_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate coordinate-level rows to one row per (sample_id, probe_id).

    For each probe j: m_j = sum(m_k), u_j = sum(u_k), n_j = m_j + u_j,
    beta_j = m_j / n_j, over all source rows k with non-missing counts.
    Rows whose counts are missing still count toward ``n_source_rows`` but
    contribute no evidence. A probe with zero rows carrying valid evidence
    gets missing aggregate coverage/fraction/counts (never coerced to 0).

    Requires and preserves total coverage: the sum of ``unmethylated_count +
    methylated_count`` over valid coordinate-level rows equals the sum of
    ``coverage`` over the resulting probe-level rows (see
    ``validation.validate_probe_aggregation``).
    """
    required = {"sample_id", "chrom", "start", "end", "probe_id", "methylated_count", "unmethylated_count"}
    missing = required - set(coord_df.columns)
    if missing:
        raise ValueError(f"aggregate_to_probe_level: input is missing columns {sorted(missing)}")

    if coord_df.empty:
        return pd.DataFrame(columns=PROBE_COLUMNS)

    df = coord_df.copy()
    has_evidence = df["methylated_count"].notna() & df["unmethylated_count"].notna()
    df["_m_valid"] = df["methylated_count"].where(has_evidence)
    df["_u_valid"] = df["unmethylated_count"].where(has_evidence)
    df["_coord"] = list(zip(df["chrom"], df["start"], df["end"]))

    grouped = df.groupby(["sample_id", "probe_id"], sort=False, dropna=False)

    agg = grouped.agg(
        methylated_count=("_m_valid", "sum"),
        unmethylated_count=("_u_valid", "sum"),
        n_source_rows=("probe_id", "size"),
        n_evidence_rows=("_m_valid", "count"),
    ).reset_index()
    n_unique_coordinates = grouped["_coord"].nunique().reset_index(name="n_unique_coordinates")
    agg = agg.merge(n_unique_coordinates, on=["sample_id", "probe_id"])

    no_evidence = agg["n_evidence_rows"] == 0
    agg["methylated_count"] = agg["methylated_count"].astype("Int64")
    agg["unmethylated_count"] = agg["unmethylated_count"].astype("Int64")
    agg.loc[no_evidence, "methylated_count"] = pd.NA
    agg.loc[no_evidence, "unmethylated_count"] = pd.NA

    agg["coverage"] = (agg["methylated_count"] + agg["unmethylated_count"]).astype("Int64")
    fraction = agg["methylated_count"].astype("Float64") / agg["coverage"].astype("Float64")
    agg["methylation_fraction"] = fraction.astype("float64")

    agg["n_source_rows"] = agg["n_source_rows"].astype("Int64")
    agg["n_unique_coordinates"] = agg["n_unique_coordinates"].astype("Int64")

    return agg[PROBE_COLUMNS].sort_values(["sample_id", "probe_id"]).reset_index(drop=True)
