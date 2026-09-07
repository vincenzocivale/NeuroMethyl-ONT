import pandas as pd
import pytest

from neuromethyl_ont.evaluation.sturgeon_adapter import (
    STURGEON_BED_COLUMNS,
    probe_level_to_sturgeon_bed,
)


def test_thresholds_and_dead_zone():
    probe_df = pd.DataFrame(
        {
            "probe_id": ["cg1", "cg2", "cg3", "cg4", "cg5"],
            "methylation_fraction": [0.9, 0.1, 0.5, 0.71, 0.29],
        }
    )

    bed = probe_level_to_sturgeon_bed(probe_df)

    assert list(bed.columns) == STURGEON_BED_COLUMNS
    # cg3 (0.5) is in the [0.3, 0.7] dead zone -> dropped
    assert set(bed["probe_id"]) == {"cg1", "cg2", "cg4", "cg5"}
    calls = dict(zip(bed["probe_id"], bed["methylation_call"], strict=True))
    assert calls["cg1"] == 1  # > 0.7
    assert calls["cg4"] == 1  # > 0.7
    assert calls["cg2"] == 0  # < 0.3
    assert calls["cg5"] == 0  # < 0.3


def test_missing_fraction_dropped():
    probe_df = pd.DataFrame({"probe_id": ["cg1", "cg2"], "methylation_fraction": [0.9, None]})

    bed = probe_level_to_sturgeon_bed(probe_df)

    assert list(bed["probe_id"]) == ["cg1"]


def test_boundary_values_excluded():
    probe_df = pd.DataFrame(
        {"probe_id": ["at_pos", "at_neg"], "methylation_fraction": [0.7, 0.3]}
    )

    bed = probe_level_to_sturgeon_bed(probe_df)

    # exactly at the threshold is not "> 0.7" or "< 0.3" -- matches Sturgeon's
    # own strict inequalities (callmapping.py: bin_scores[scores > pos] = 1)
    assert bed.empty


def test_custom_thresholds():
    probe_df = pd.DataFrame({"probe_id": ["cg1"], "methylation_fraction": [0.6]})

    bed = probe_level_to_sturgeon_bed(probe_df, pos_threshold=0.5, neg_threshold=0.4)

    assert list(bed["probe_id"]) == ["cg1"]
    assert bed["methylation_call"].iloc[0] == 1


def test_rejects_invalid_thresholds():
    probe_df = pd.DataFrame({"probe_id": ["cg1"], "methylation_fraction": [0.6]})
    with pytest.raises(ValueError):
        probe_level_to_sturgeon_bed(probe_df, pos_threshold=0.3, neg_threshold=0.7)


def test_empty_input():
    probe_df = pd.DataFrame({"probe_id": [], "methylation_fraction": []})
    bed = probe_level_to_sturgeon_bed(probe_df)
    assert bed.empty
    assert list(bed.columns) == STURGEON_BED_COLUMNS
