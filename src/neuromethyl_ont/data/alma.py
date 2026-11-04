"""Reader for the ALMA acute-leukemia methylation atlas (Marchi et al. 2025,
doi:10.1038/s41467-025-62005-4) -- see docs/ALMA_MILESTONE.md.

This is a separate classifier milestone from the CNS tumor pipeline: ALMA
covers acute leukemias (AML/ALL/MDS/APL/MPAL), not CNS tumors, and has its
own reference cohort and class taxonomy. Nothing here feeds the GSE90496-derived
91-class CNS models (Binary/Beta/Evidence).

Two public files, both under ``datasets/ALMA/raw/`` (GitHub f-marchi/ALMA
v0.2.0 release):

- ``3314samples_331556cpgs_bvalues.pkl.gz``: discovery/training array betas,
  index = sample_id, columns = ``Batch`` + ~331556 ``cg...`` probe columns.
- ``20samples_nanopore_866857cpgs.csv.gz``: ONT evaluation cohort, CpGs as
  rows, samples (``alma_<n>_<BM|PB>``) as columns -- the transpose of the
  discovery layout.

Clinical labels (Nature Communications supplementary file MOESM4,
``datasets/ALMA/raw/clinical_data_MOESM4.xlsx``, sheet ``indexed_clinical_data``)
join to the discovery betas by ``sample_id`` -- the two files were verified
to agree on every row's ``Batch`` value (0/3314 mismatches). Row *order*
does not correspond between the two files (the published clinical sheet is
row-shuffled, evidently for de-identification), so this join must always be
by index, never by position.

No equivalent label file has been found for the 20 nanopore samples: their
``alma_<n>_<BM|PB>`` identifiers do not appear anywhere in the clinical
Excel (all sheets, including the figure-data tabs) or in the authors'
published notebooks. :func:`load_nanopore_betas` therefore returns betas
only, with no label column, until those labels are recovered (e.g. by
transcribing the article's own patient table).
"""

from __future__ import annotations

import gzip
import pickle
from pathlib import Path

import pandas as pd

from neuromethyl_ont.data.paths import get_data_root

_DATASET_DIRNAME = "ALMA"
_DISCOVERY_BETA_FILENAME = "3314samples_331556cpgs_bvalues.pkl.gz"
_NANOPORE_BETA_FILENAME = "20samples_nanopore_866857cpgs.csv.gz"
_CLINICAL_LABELS_FILENAME = "discovery_clinical_labels.parquet"

CLINICAL_LABEL_COLUMNS = [
    "sample_id",
    "Batch",
    "WHO 2022 Diagnosis",
    "AL Epigenomic Subtype",
    "AL Epigenomic Subtype_int",
    "Train-Test",
    "Sex",
    "Patient_ID",
    "Age (years)",
    "Tissue Type",
    "Sample Type",
]


def _alma_root(data_root: Path | None = None) -> Path:
    root = data_root if data_root is not None else get_data_root()
    return root / "datasets" / _DATASET_DIRNAME


def load_discovery_betas(data_root: Path | None = None) -> pd.DataFrame:
    """Load the 3314-sample discovery beta matrix (index = sample_id).

    Includes the ``Batch`` column shipped alongside the probe columns; the
    caller is responsible for excluding it before treating the frame as a
    pure feature matrix.
    """
    path = _alma_root(data_root) / "raw" / _DISCOVERY_BETA_FILENAME
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def load_discovery_labels(data_root: Path | None = None) -> pd.DataFrame:
    """Load the discovery cohort's clinical labels, indexed by sample_id.

    Produced by joining :func:`load_discovery_betas` to the Nature
    Communications clinical Excel by ``sample_id`` (see module docstring).
    Not all rows have an ``AL Epigenomic Subtype`` (2987/3314 do; the rest
    are unlabeled/excluded in the source publication).
    """
    path = _alma_root(data_root) / "processed" / _CLINICAL_LABELS_FILENAME
    labels = pd.read_parquet(path)
    return labels.set_index("sample_id")


def load_discovery_dataset(data_root: Path | None = None) -> pd.DataFrame:
    """Load discovery betas joined with clinical labels, indexed by sample_id.

    Raises if any row's ``Batch`` disagrees between the beta matrix and the
    clinical labels, since that would indicate the two files have drifted
    out of sync since the join was last verified.
    """
    betas = load_discovery_betas(data_root)
    labels = load_discovery_labels(data_root)
    joined = betas.join(labels, how="left", rsuffix="_clin")
    mismatched = joined["Batch"] != joined["Batch_clin"]
    if mismatched.any():
        bad = joined.index[mismatched].tolist()[:5]
        raise ValueError(
            f"ALMA discovery beta/label join disagrees on Batch for "
            f"{mismatched.sum()} sample(s), e.g. {bad}"
        )
    return joined.drop(columns=["Batch_clin"])


def load_nanopore_betas(data_root: Path | None = None) -> pd.DataFrame:
    """Load the 20-sample ONT evaluation cohort, transposed to sample-indexed rows.

    The source CSV ships CpGs as rows and samples as columns; this returns
    the transpose (index = sample_id, columns = probe_id) to match the
    discovery layout. No clinical labels are attached -- see module
    docstring.
    """
    path = _alma_root(data_root) / "raw" / _NANOPORE_BETA_FILENAME
    betas = pd.read_csv(path, index_col=0)
    return betas.transpose()
