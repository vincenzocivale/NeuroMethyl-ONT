import numpy as np
import pandas as pd
import pytest
from torch import nn

from neuromethyl_ont.inference.two_stage import (
    UNDETERMINED,
    decide_domain,
    load_counts_aligned,
    route,
)


def test_load_counts_aligned_fills_missing_probes_with_zero(tmp_path):
    path = tmp_path / "c.tsv"
    pd.DataFrame(
        {"probe_id": ["a", "c"], "methylated_count": [3, 1], "unmethylated_count": [1, 4]}
    ).to_csv(path, sep="\t", index=False)
    m, u = load_counts_aligned(path, ["a", "b", "c"])
    assert m.tolist() == [3.0, 0.0, 1.0]
    assert u.tolist() == [1.0, 0.0, 4.0]


def test_load_counts_aligned_rejects_duplicate_probe_rows(tmp_path):
    path = tmp_path / "c.tsv"
    pd.DataFrame(
        {"probe_id": ["a", "a"], "methylated_count": [1, 1], "unmethylated_count": [1, 1]}
    ).to_csv(path, sep="\t", index=False)
    with pytest.raises(ValueError, match="unique"):
        load_counts_aligned(path, ["a"])


def test_decide_domain_abstains_below_threshold():
    assert decide_domain(np.array([0.95, 0.05]), 0.9) == "cns"
    assert decide_domain(np.array([0.05, 0.95]), 0.9) == "leukemia"
    assert decide_domain(np.array([0.7, 0.3]), 0.9) == UNDETERMINED


def test_route_ignores_probes_outside_the_shared_mask():
    router = nn.Linear(4, 2)
    nn.init.zeros_(router.weight)
    nn.init.zeros_(router.bias)
    router.weight.data[1, 0] = 50.0  # only probe 0 pushes toward leukemia
    mask = np.array([False, True, True, True])
    m = np.array([9.0, 0.0, 0.0, 0.0])
    u = np.array([0.0, 0.0, 0.0, 0.0])
    proba = route(router, mask, m, u)
    assert proba.shape == (2,) and abs(proba.sum() - 1.0) < 1e-6
    assert abs(proba[0] - 0.5) < 1e-6  # masked probe 0 carries no signal
