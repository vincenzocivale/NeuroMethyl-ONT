"""Canonical schemas for Nanopore/array methylation records.

Two canonical representations are defined (see docs/DATA_LAYOUT.md and the
project brief):

- ``CoordinateMethylationRecord``: one row per genomic observation
  (sample_id + chrom + start + end). ``probe_id`` is annotation only.
- ``ProbeMethylationRecord``: one row per (sample_id, probe_id), produced by
  evidence-weighted aggregation of coordinate-level rows (sum of methylated /
  unmethylated read counts, never a plain average of betas).

Column order below is the on-disk column order for Parquet outputs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Probe IDs are not guaranteed to be Illumina cg probes (e.g. "MGMT", "rs..."
# SNP control probes are observed in Rapid-CNS2 outputs). This pattern is used
# only for QC/logging purposes -- it must never be used to filter rows.
CG_PROBE_PATTERN = re.compile(r"^cg\d+$")

COORDINATE_COLUMNS = [
    "sample_id",
    "chrom",
    "start",
    "end",
    "probe_id",
    "coverage",
    "methylation_fraction",
    "methylated_count",
    "unmethylated_count",
]

PROBE_COLUMNS = [
    "sample_id",
    "probe_id",
    "coverage",
    "methylation_fraction",
    "methylated_count",
    "unmethylated_count",
    "n_source_rows",
    "n_unique_coordinates",
]

# pandas dtypes for the numeric columns shared by both representations.
# Nullable integer dtypes are used so that missing methylation is representable
# as <NA> rather than silently coerced to 0 or NaN-as-float truncation.
COORDINATE_DTYPES = {
    "sample_id": "string",
    "chrom": "string",
    "start": "int64",
    "end": "int64",
    "probe_id": "string",
    "coverage": "Int64",
    "methylation_fraction": "float64",
    "methylated_count": "Int64",
    "unmethylated_count": "Int64",
}

PROBE_DTYPES = {
    "sample_id": "string",
    "probe_id": "string",
    "coverage": "Int64",
    "methylation_fraction": "float64",
    "methylated_count": "Int64",
    "unmethylated_count": "Int64",
    "n_source_rows": "Int64",
    "n_unique_coordinates": "Int64",
}


@dataclass(frozen=True)
class CoordinateMethylationRecord:
    """One genomic observation. Primary key: (sample_id, chrom, start, end).

    ``probe_id`` is annotation, not identity: the same coordinate can carry
    more than one probe annotation, and the same probe_id can be annotated at
    more than one coordinate (see docs on MNP-Flex bedtools-intersect
    duplication).
    """

    sample_id: str
    chrom: str
    start: int
    end: int
    probe_id: str
    coverage: int | None
    methylation_fraction: float | None
    methylated_count: int | None
    unmethylated_count: int | None


@dataclass(frozen=True)
class ProbeMethylationRecord:
    """One (sample_id, probe_id) aggregate, evidence-weighted over all
    coordinate-level rows sharing that probe_id within the sample.
    """

    sample_id: str
    probe_id: str
    coverage: int | None
    methylation_fraction: float | None
    methylated_count: int | None
    unmethylated_count: int | None
    n_source_rows: int
    n_unique_coordinates: int
