"""Convert this project's canonical probe-level methylation to Sturgeon's own
input format, without going through Sturgeon's ``inputtobed`` (which expects
raw BAM/modkit output and a genome alignment this project's canonical
outputs don't carry).

Read from Sturgeon's own source (``sturgeon/prediction.py``,
``sturgeon/callmapping.py`` in the installed ``sturgeon==0.5.1`` package):
its ``predict`` step only ever consumes a whitespace-delimited bed file with
columns ``chrom chromStart chromEnd methylation_call probe_id`` and
reindexes strictly by ``probe_id`` (a cg-number) against the model's own
``probes.csv`` -- genomic coordinates are read but never used for matching.
``methylation_call`` is binary (0 = unmethylated evidence, 1 = methylated
evidence); a probe absent from the bed file becomes Sturgeon's "no measure"
value on load. ``chrom``/``chromStart``/``chromEnd`` are therefore written
as placeholders here (never read by prediction code) -- see
``docs/DATA_LAYOUT.md`` for why canonical probe-level parquet has no
per-probe genomic coordinates to put there instead.

Sturgeon derives that binary call itself (``modkit_pileup_file_to_bed`` /
``map_methyl_calls_to_probes_chr``) by thresholding each covered position's
fraction-methylated at ``pos_threshold``/``neg_threshold`` (default
0.7/0.3, values in between excluded as ambiguous) and taking the median
across positions within a probe's margin. This project's canonical
probe-level ``methylation_fraction`` is already a single aggregate value per
probe (summed methylated/unmethylated counts across duplicate coordinates,
see ``canonical.aggregate_to_probe_level``), so :func:`probe_level_to_sturgeon_bed`
applies that same threshold directly to the aggregate fraction -- the
closest faithful reproduction of Sturgeon's own decision rule given how this
project's evidence is already aggregated, not a byte-identical replay of
their per-position pileup path.
"""

from __future__ import annotations

import pandas as pd

DEFAULT_POS_THRESHOLD = 0.7
DEFAULT_NEG_THRESHOLD = 0.3

STURGEON_BED_COLUMNS = ["chrom", "chromStart", "chromEnd", "methylation_call", "probe_id"]


def probe_level_to_sturgeon_bed(
    probe_df: pd.DataFrame,
    pos_threshold: float = DEFAULT_POS_THRESHOLD,
    neg_threshold: float = DEFAULT_NEG_THRESHOLD,
) -> pd.DataFrame:
    """One sample's canonical probe-level rows -> a Sturgeon-format bed frame.

    ``probe_df`` must have ``probe_id`` and ``methylation_fraction`` columns
    (as written by ``canonical.aggregate_to_probe_level``/the RapidCNS2 and
    GSE289246 canonical parquet outputs). Rows with a missing fraction, or a
    fraction in ``(neg_threshold, pos_threshold]`` -- ambiguous, matching
    Sturgeon's own dead zone -- are dropped, never guessed. Returns columns
    :data:`STURGEON_BED_COLUMNS`, ready for ``load_bed_file``/``bed_to_numpy``.
    """
    if pos_threshold <= neg_threshold:
        raise ValueError(
            f"pos_threshold ({pos_threshold}) must be > neg_threshold ({neg_threshold})"
        )

    df = probe_df[["probe_id", "methylation_fraction"]].dropna(subset=["methylation_fraction"])

    methylated = df["methylation_fraction"] > pos_threshold
    unmethylated = df["methylation_fraction"] < neg_threshold
    df = df[methylated | unmethylated].copy()

    out = pd.DataFrame(
        {
            "chrom": "0",
            "chromStart": 0,
            "chromEnd": 1,
            "methylation_call": methylated.loc[df.index].astype(int),
            "probe_id": df["probe_id"],
        }
    )
    return out[STURGEON_BED_COLUMNS].reset_index(drop=True)
