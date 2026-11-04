"""Patient-level grouping and fixed-seed grouped folds for the MethyLYZR cohort."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from neuromethyl_ont.evaluation.taxonomy_mapping import coarse_truth

MANIFEST_COLUMNS = [
    "cohort", "file", "sample_id", "run_hash", "patient_id", "label_short", "coarse_label",
    "is_metastasis", "abstention_test", "split_group", "patient_in_both_R9_R10",
    "is_repeat_suffix_2",
]


def build_manifest(labels: pd.DataFrame, taxonomy: dict) -> pd.DataFrame:
    """Minimal MethyLYZR label table -> manifest with coarse label and ``split_group`` = patient_id.

    Age, sex and site are never carried over. Raises if a patient has conflicting labels.
    """
    df = labels.copy()
    coarse = [coarse_truth(x, taxonomy) for x in df["label_short"]]
    df["coarse_label"] = [c for c, _ in coarse]
    df["abstention_test"] = [int(a) for _, a in coarse]
    df["split_group"] = df["patient_id"]
    conflicts = df.groupby("split_group")["label_short"].nunique()
    if (conflicts > 1).any():
        raise ValueError(f"patients with conflicting labels: {sorted(conflicts[conflicts > 1].index)}")
    return df[MANIFEST_COLUMNS].reset_index(drop=True)


def grouped_folds(
    groups, y=None, n_splits: int = 5, seed: int = 0
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Deterministic (train_idx, test_idx) folds in which no group is split across folds.

    ``y`` (optional) stratifies by class as far as the groups allow.
    """
    groups = np.asarray(groups)
    y = np.zeros(len(groups), dtype=int) if y is None else np.asarray(y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # rare classes have fewer groups than folds
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(cv.split(np.zeros(len(groups)), y, groups))
