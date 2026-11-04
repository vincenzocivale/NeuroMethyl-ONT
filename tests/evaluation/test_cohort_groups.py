from pathlib import Path

import pandas as pd
import pytest

from neuromethyl_ont.evaluation.cohort_groups import build_manifest, grouped_folds
from neuromethyl_ont.evaluation.taxonomy_mapping import load_taxonomy

ROOT = Path(__file__).resolve().parents[2]
TAX = load_taxonomy(ROOT / "configs" / "taxonomies" / "methylyzr_who_level.yaml")
MANIFEST = pd.read_csv(ROOT / "registry" / "methylyzr_cohort_groups.tsv", sep="\t", dtype=str)


def test_committed_manifest_shape_and_no_demographics():
    assert len(MANIFEST) == 282 and MANIFEST["split_group"].nunique() == 192
    assert not {"age", "sex", "site"} & set(MANIFEST.columns)


def test_metastasis_rows_flagged():
    mets = MANIFEST[MANIFEST["is_metastasis"] == "1"]
    assert len(mets) == 27
    assert set(mets["abstention_test"]) == {"1"} and set(mets["coarse_label"]) == {"OUT_OF_SCOPE"}
    assert (MANIFEST[MANIFEST["is_metastasis"] == "0"]["abstention_test"] == "0").all()


def test_cross_cohort_patients_share_one_group():
    both = MANIFEST[MANIFEST["patient_in_both_R9_R10"] == "1"]
    assert both["split_group"].nunique() == 33
    assert (both.groupby("split_group")["cohort"].nunique() == 2).all()
    # every repeat library and run of a patient carries the same group
    assert (MANIFEST["split_group"] == MANIFEST["patient_id"]).all()


def test_build_manifest_rejects_conflicting_patient_labels():
    df = pd.DataFrame({
        "cohort": ["R9", "R10"], "file": ["a", "b"], "sample_id": ["IEG1", "IEG1_2"], "run_hash": ["", ""],
        "patient_id": ["IEG1", "IEG1"], "label_short": ["GBM", "MB"], "is_metastasis": [0, 0],
        "patient_in_both_R9_R10": [1, 1], "is_repeat_suffix_2": [0, 1],
    })
    with pytest.raises(ValueError):
        build_manifest(df, TAX)


def test_grouped_folds_keep_groups_together_and_are_deterministic():
    groups = MANIFEST["split_group"].to_numpy()
    y = MANIFEST["coarse_label"].to_numpy()
    folds = grouped_folds(groups, y, n_splits=5, seed=7)
    assert len(folds) == 5
    seen = set()
    for train, test in folds:
        assert not set(groups[train]) & set(groups[test])
        seen |= set(test)
    assert seen == set(range(len(groups)))
    again = grouped_folds(groups, y, n_splits=5, seed=7)
    assert all((a[1] == b[1]).all() for a, b in zip(folds, again, strict=True))
