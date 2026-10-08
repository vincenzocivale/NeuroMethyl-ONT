import gzip
import pickle

import pandas as pd
import pytest

from neuromethyl_ont.data.alma import load_discovery_dataset, load_nanopore_betas
from neuromethyl_ont.data.paths import get_data_root


def _write_discovery(data_root, batches):
    raw_dir = data_root / "datasets" / "ALMA" / "raw"
    raw_dir.mkdir(parents=True)
    betas = pd.DataFrame(
        {"Batch": batches, "cg00000001": [0.1] * len(batches)},
        index=[f"sample_{i}" for i in range(len(batches))],
    )
    with gzip.open(raw_dir / "3314samples_331556cpgs_bvalues.pkl.gz", "wb") as f:
        pickle.dump(betas, f)
    return betas


def _write_labels(data_root, batches, subtypes):
    processed_dir = data_root / "datasets" / "ALMA" / "processed"
    processed_dir.mkdir(parents=True)
    labels = pd.DataFrame(
        {
            "sample_id": [f"sample_{i}" for i in range(len(batches))],
            "Batch": batches,
            "AL Epigenomic Subtype": subtypes,
        }
    )
    labels.to_parquet(processed_dir / "discovery_clinical_labels.parquet", index=False)
    return labels


def test_load_discovery_dataset_joins_by_sample_id(tmp_path):
    _write_discovery(tmp_path, ["GSE49031", "GDC_TARGET-AML"])
    _write_labels(tmp_path, ["GSE49031", "GDC_TARGET-AML"], ["T-ALL NOS", None])

    joined = load_discovery_dataset(data_root=tmp_path)

    assert list(joined.index) == ["sample_0", "sample_1"]
    assert joined.loc["sample_0", "AL Epigenomic Subtype"] == "T-ALL NOS"
    assert pd.isna(joined.loc["sample_1", "AL Epigenomic Subtype"])
    assert "Batch_clin" not in joined.columns


def test_load_discovery_dataset_raises_on_batch_mismatch(tmp_path):
    _write_discovery(tmp_path, ["GSE49031"])
    _write_labels(tmp_path, ["GDC_TARGET-AML"], ["AML with mutated NPM1"])

    with pytest.raises(ValueError, match="Batch"):
        load_discovery_dataset(data_root=tmp_path)


@pytest.mark.skipif(
    get_data_root(required=False) is None,
    reason="requires NEUROMETHYL_DATA_ROOT",
)
def test_real_discovery_dataset_has_no_batch_mismatch():
    root = get_data_root()
    raw_path = root / "datasets" / "ALMA" / "raw" / "3314samples_331556cpgs_bvalues.pkl.gz"
    if not raw_path.exists():
        pytest.skip("ALMA discovery beta file not present under NEUROMETHYL_DATA_ROOT")

    joined = load_discovery_dataset()

    assert len(joined) == 3314
    assert joined["AL Epigenomic Subtype"].notna().sum() == 2987


@pytest.mark.skipif(
    get_data_root(required=False) is None,
    reason="requires NEUROMETHYL_DATA_ROOT",
)
def test_real_nanopore_betas_shape():
    root = get_data_root()
    raw_path = root / "datasets" / "ALMA" / "raw" / "20samples_nanopore_866857cpgs.csv.gz"
    if not raw_path.exists():
        pytest.skip("ALMA nanopore beta file not present under NEUROMETHYL_DATA_ROOT")

    betas = load_nanopore_betas()

    assert betas.shape[0] == 20
