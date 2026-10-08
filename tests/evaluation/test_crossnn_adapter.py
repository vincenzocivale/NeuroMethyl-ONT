import pandas as pd

from neuromethyl_ont.evaluation.crossnn_adapter import (
    CROSSNN_BED_COLUMNS,
    probe_level_to_crossnn_bed,
)


def test_single_threshold_no_dead_zone():
    probe_df = pd.DataFrame(
        {
            "probe_id": ["cg1", "cg2", "cg3", "cg4", "cg5"],
            "methylation_fraction": [0.9, 0.1, 0.5, 0.71, 0.29],
        }
    )

    bed = probe_level_to_crossnn_bed(probe_df)

    assert list(bed.columns) == CROSSNN_BED_COLUMNS
    # unlike Sturgeon's dead zone, every observed probe gets a call -- cg3
    # (0.5) is not dropped.
    assert set(bed["probe_id"]) == {"cg1", "cg2", "cg3", "cg4", "cg5"}
    calls = dict(zip(bed["probe_id"], bed["methylation_call"], strict=True))
    assert calls["cg1"] == 1  # >= 0.6
    assert calls["cg4"] == 1  # >= 0.6
    assert calls["cg3"] == 0  # < 0.6
    assert calls["cg2"] == 0  # < 0.6
    assert calls["cg5"] == 0  # < 0.6


def test_boundary_value_is_methylated():
    # crossNN's own rule is "< 60 -> -1, >= 60 -> +1" (NN_model.py
    # predict_from_bedMethyl): exactly at the threshold counts as methylated.
    probe_df = pd.DataFrame({"probe_id": ["cg1"], "methylation_fraction": [0.6]})

    bed = probe_level_to_crossnn_bed(probe_df)

    assert list(bed["probe_id"]) == ["cg1"]
    assert bed["methylation_call"].iloc[0] == 1


def test_missing_fraction_dropped():
    probe_df = pd.DataFrame({"probe_id": ["cg1", "cg2"], "methylation_fraction": [0.9, None]})

    bed = probe_level_to_crossnn_bed(probe_df)

    assert list(bed["probe_id"]) == ["cg1"]


def test_custom_threshold():
    probe_df = pd.DataFrame(
        {"probe_id": ["cg1", "cg2"], "methylation_fraction": [0.55, 0.45]}
    )

    bed = probe_level_to_crossnn_bed(probe_df, threshold=0.5)

    calls = dict(zip(bed["probe_id"], bed["methylation_call"], strict=True))
    assert calls["cg1"] == 1
    assert calls["cg2"] == 0


def test_empty_input():
    probe_df = pd.DataFrame({"probe_id": [], "methylation_fraction": []})
    bed = probe_level_to_crossnn_bed(probe_df)
    assert bed.empty
    assert list(bed.columns) == CROSSNN_BED_COLUMNS
