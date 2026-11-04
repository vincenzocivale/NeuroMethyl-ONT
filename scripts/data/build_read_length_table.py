"""Save the pooled read-length quantile table of extracted MethyLYZR ONT_R10 feather files.

Usage: NEUROMETHYL_DATA_ROOT=... python -I build_read_length_table.py <dir with *.feather> <out.json>
(extract the archive into a fresh scratch directory first; the files are untrusted data).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from neuromethyl_ont.data.array_reference import load_feature_index  # noqa: E402
from neuromethyl_ont.data.paths import get_data_root  # noqa: E402
from neuromethyl_ont.evaluation.read_coverage import (  # noqa: E402
    read_length_table,
    write_json,
)

files = sorted(Path(sys.argv[1]).glob("*.feather"))
probe_ids = load_feature_index(
    get_data_root() / "derived/reference/GSE90496/feature_index.json"
).probe_ids
table = read_length_table(
    files, probe_ids, source="MethyLYZR ONT_R10.tar.gz (Zenodo 13236097), ONT R10 PromethION"
)
table["files"] = [f.name for f in files]
write_json(table, Path(sys.argv[2]))
print(f"{table['n_reads']} reads from {len(files)} files -> {sys.argv[2]}")
