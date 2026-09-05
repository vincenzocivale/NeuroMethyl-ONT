from pathlib import Path

from neuromethyl_ont.data.paths import get_data_root


def test_get_data_root(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("NEUROMETHYL_DATA_ROOT", str(tmp_path))
    assert get_data_root() == tmp_path.resolve()
