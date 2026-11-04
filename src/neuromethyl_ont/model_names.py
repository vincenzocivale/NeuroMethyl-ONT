"""Display names of the models and the mapping from the legacy names.

Binary, Beta, Evidence and Consensus were called B0, C0, E1 and B0+E1 in older
saved results (docs/MODEL_NAMES.md). Every reader of saved tables passes them
through ``canonicalize_frame`` / ``read_results_tsv``; writers use the new names.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

LEGACY_MODEL_NAMES = {"B0": "Binary", "C0": "Beta", "E1": "Evidence", "B0+E1": "Consensus"}
_MODEL_COLUMNS = ("model", "model_a", "model_b")


def canonical_model_name(name: str) -> str:
    """The current name for ``name`` (unchanged if already current or not one of ours)."""
    return LEGACY_MODEL_NAMES.get(name, name)


def canonicalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Copy of ``df`` with legacy names mapped in the model columns' values and in the
    column labels (``B0``, ``B0_confidence``, ``B0+E1_predicted_family``)."""
    out = df.copy()
    for col in _MODEL_COLUMNS:
        if col in out.columns:
            out[col] = out[col].map(canonical_model_name)

    def rename(label):
        for old, new in LEGACY_MODEL_NAMES.items():
            if label == old or (isinstance(label, str) and label.startswith(old + "_")):
                return new + label[len(old):]
        return label

    return out.rename(columns=rename)


def read_results_tsv(path: str | Path) -> pd.DataFrame:
    """Read a saved results TSV (old or new names) with current model names."""
    return canonicalize_frame(pd.read_csv(path, sep="\t"))
