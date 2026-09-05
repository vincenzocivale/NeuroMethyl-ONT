from pathlib import Path

from neuromethyl_ont.data.registry import validate_dataset_registry


def test_dataset_registry_schema():
    root = Path(__file__).resolve().parents[1]
    assert validate_dataset_registry(root / "registry" / "datasets.tsv") == []
