from __future__ import annotations

from pathlib import Path

import pandas as pd

DATASET_REQUIRED_COLUMNS = {
    "dataset_id",
    "modality",
    "default_role",
    "accession_or_source",
    "raw_format",
    "status",
}

SAMPLE_REQUIRED_COLUMNS = {
    "global_patient_id",
    "dataset_id",
    "source_sample_id",
    "role",
    "is_duplicate",
    "include",
}


def read_tsv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def validate_columns(df: pd.DataFrame, required: set[str], name: str) -> list[str]:
    missing = sorted(required.difference(df.columns))
    return [f"{name}: missing required column '{column}'" for column in missing]


def validate_dataset_registry(path: str | Path) -> list[str]:
    df = read_tsv(path)
    errors = validate_columns(df, DATASET_REQUIRED_COLUMNS, "datasets")
    if "dataset_id" in df.columns and df["dataset_id"].duplicated().any():
        errors.append("datasets: dataset_id values must be unique")
    return errors


def validate_sample_registry(path: str | Path) -> list[str]:
    df = read_tsv(path)
    errors = validate_columns(df, SAMPLE_REQUIRED_COLUMNS, "samples")
    return errors
