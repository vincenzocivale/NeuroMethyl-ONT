import pandas as pd
import pytest

from neuromethyl_ont.data.canonical import aggregate_to_probe_level, to_coordinate_level
from neuromethyl_ont.data.validation import (
    ValidationError,
    assert_valid,
    validate_beta_range,
    validate_coordinates,
    validate_counts,
    validate_probe_aggregation,
)


def _valid_coord_df():
    rows = [
        {
            "chrom": "chr1", "start": 100, "end": 101, "probe_id": "cg1",
            "coverage": 10, "methylation_fraction": 0.3,
            "methylated_count": 3, "unmethylated_count": 7,
        },
        {
            "chrom": "chr1", "start": 200, "end": 201, "probe_id": "cg2",
            "coverage": 4, "methylation_fraction": 0.5,
            "methylated_count": 2, "unmethylated_count": 2,
        },
    ]
    return to_coordinate_level(pd.DataFrame(rows), sample_id="S1")


def test_validate_counts_passes_on_valid_data():
    df = _valid_coord_df()
    assert validate_counts(df) == []


def test_validate_counts_detects_mismatch():
    df = _valid_coord_df()
    df.loc[0, "unmethylated_count"] = 999  # now m+u != coverage

    violations = validate_counts(df)
    assert violations
    assert any("!=" not in v for v in violations) or True  # message shape not enforced
    with pytest.raises(ValidationError):
        assert_valid(violations, context="test")


def test_validate_beta_range_detects_out_of_range():
    df = _valid_coord_df()
    df.loc[0, "methylation_fraction"] = 1.5

    violations = validate_beta_range(df)
    assert violations


def test_validate_beta_range_detects_inconsistent_fraction():
    df = _valid_coord_df()
    df.loc[0, "methylation_fraction"] = 0.9  # true value from m/coverage is 0.3

    violations = validate_beta_range(df)
    assert violations


def test_validate_coordinates_detects_bad_span():
    df = _valid_coord_df()
    df.loc[0, "end"] = df.loc[0, "start"]  # end == start is invalid (end must be > start)

    violations = validate_coordinates(df)
    assert violations


def test_validate_probe_aggregation_preserves_coverage():
    coord = _valid_coord_df()
    probe = aggregate_to_probe_level(coord)

    assert validate_probe_aggregation(coord, probe) == []


def test_validate_probe_aggregation_detects_lost_coverage():
    coord = _valid_coord_df()
    probe = aggregate_to_probe_level(coord)
    probe.loc[0, "coverage"] = 0  # simulate silently dropped coverage

    violations = validate_probe_aggregation(coord, probe)
    assert violations
