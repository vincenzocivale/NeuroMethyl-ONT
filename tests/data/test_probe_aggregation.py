import math

import pandas as pd

from neuromethyl_ont.data.canonical import aggregate_to_probe_level, to_coordinate_level


def _coord_df(rows):
    df = pd.DataFrame(rows)
    return to_coordinate_level(df, sample_id="S1")


def test_probe_aggregation_sums_evidence_not_average_of_betas():
    # cgX row1: n=2, beta=.5 -> m=1,u=1 ; row2: n=1, beta=1 -> m=1,u=0
    coord = _coord_df(
        [
            {
                "chrom": "chr1",
                "start": 100,
                "end": 101,
                "probe_id": "cgX",
                "coverage": 2,
                "methylation_fraction": 0.5,
                "methylated_count": 1,
                "unmethylated_count": 1,
            },
            {
                "chrom": "chr1",
                "start": 200,
                "end": 201,
                "probe_id": "cgX",
                "coverage": 1,
                "methylation_fraction": 1.0,
                "methylated_count": 1,
                "unmethylated_count": 0,
            },
        ]
    )

    probe = aggregate_to_probe_level(coord)

    assert len(probe) == 1
    row = probe.iloc[0]
    assert row["probe_id"] == "cgX"
    assert row["methylated_count"] == 2
    assert row["unmethylated_count"] == 1
    assert row["coverage"] == 3
    assert math.isclose(row["methylation_fraction"], 2 / 3)
    assert row["n_source_rows"] == 2
    assert row["n_unique_coordinates"] == 2

    # A naive mean of betas would give (0.5 + 1.0) / 2 = 0.75, which must NOT
    # be what this function produces.
    assert not math.isclose(row["methylation_fraction"], 0.75)


def test_probe_aggregation_preserves_total_coverage():
    coord = _coord_df(
        [
            {
                "chrom": "chr1", "start": 1, "end": 2, "probe_id": "cgA",
                "coverage": 10, "methylation_fraction": 0.3,
                "methylated_count": 3, "unmethylated_count": 7,
            },
            {
                "chrom": "chr1", "start": 2, "end": 3, "probe_id": "cgA",
                "coverage": 5, "methylation_fraction": 0.6,
                "methylated_count": 3, "unmethylated_count": 2,
            },
            {
                "chrom": "chr2", "start": 1, "end": 2, "probe_id": "cgB",
                "coverage": 8, "methylation_fraction": 0.0,
                "methylated_count": 0, "unmethylated_count": 8,
            },
        ]
    )

    probe = aggregate_to_probe_level(coord)

    assert coord["coverage"].sum() == probe["coverage"].sum() == 23


def test_probe_aggregation_missing_evidence_stays_missing_not_zero():
    coord = _coord_df(
        [
            {
                "chrom": "chr1", "start": 1, "end": 2, "probe_id": "cgMissing",
                "coverage": pd.NA, "methylation_fraction": float("nan"),
                "methylated_count": pd.NA, "unmethylated_count": pd.NA,
            },
        ]
    )

    probe = aggregate_to_probe_level(coord)

    assert len(probe) == 1
    row = probe.iloc[0]
    assert pd.isna(row["coverage"])
    assert pd.isna(row["methylated_count"])
    assert pd.isna(row["unmethylated_count"])
    assert row["n_source_rows"] == 1


def test_probe_aggregation_partial_evidence_only_sums_valid_rows():
    coord = _coord_df(
        [
            {
                "chrom": "chr1", "start": 1, "end": 2, "probe_id": "cgP",
                "coverage": 4, "methylation_fraction": 0.5,
                "methylated_count": 2, "unmethylated_count": 2,
            },
            {
                "chrom": "chr1", "start": 2, "end": 3, "probe_id": "cgP",
                "coverage": pd.NA, "methylation_fraction": float("nan"),
                "methylated_count": pd.NA, "unmethylated_count": pd.NA,
            },
        ]
    )

    probe = aggregate_to_probe_level(coord)

    row = probe.iloc[0]
    assert row["coverage"] == 4
    assert row["methylated_count"] == 2
    assert row["unmethylated_count"] == 2
    assert row["n_source_rows"] == 2
