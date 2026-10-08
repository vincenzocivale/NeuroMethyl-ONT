"""Patient-disjoint, subtype-stratified train/val/test split for ALMA discovery.

The public ALMA release marks every labelled sample as ``Train Sample`` (no
held-out test), and 2987 labelled samples come from only 2776 patients, so a
sample-level split would leak repeated patients across splits. Samples are
grouped by ``Patient_ID`` and stratified by subtype (docs/EXPERIMENT_POLICY.md).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

SPLITS = ("train", "val", "test")


def make_patient_split(
    frame: pd.DataFrame,
    label_col: str,
    patient_col: str,
    seed: int = 17,
    holdout_folds: tuple[int, int] = (7, 6),
) -> pd.Series:
    """Return a ``split`` label (train/val/test) per row of ``frame``.

    Test takes ~1/holdout_folds[0] of the patients, then val takes
    ~1/holdout_folds[1] of the rest. Every patient lands in exactly one split.
    """
    labels = frame[label_col].to_numpy()
    groups = frame[patient_col].to_numpy()
    split = pd.Series("train", index=frame.index, name="split")

    def take_fold(mask: np.ndarray, n_splits: int) -> np.ndarray:
        idx = np.flatnonzero(mask)
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        _, fold = next(sgkf.split(idx, labels[idx], groups[idx]))
        return idx[fold]

    remaining = np.ones(len(frame), dtype=bool)
    for name, n_splits in zip(("test", "val"), holdout_folds, strict=True):
        chosen = take_fold(remaining, n_splits)
        split.iloc[chosen] = name
        remaining[chosen] = False
    return split
