from pathlib import Path

import pytest

from neuromethyl_ont.data.paths import get_data_root


def test_get_data_root(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("NEUROMETHYL_DATA_ROOT", str(tmp_path))
    assert get_data_root() == tmp_path.resolve()


def test_get_data_root_missing_raises(monkeypatch):
    monkeypatch.delenv("NEUROMETHYL_DATA_ROOT", raising=False)
    with pytest.raises(RuntimeError, match="NEUROMETHYL_DATA_ROOT"):
        get_data_root()
    assert get_data_root(required=False) is None
