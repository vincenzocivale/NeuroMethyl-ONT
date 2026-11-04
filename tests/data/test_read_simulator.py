"""Read-structured simulator (R13): determinism, count invariants, clusters, Poisson unchanged."""

from pathlib import Path

import numpy as np
import pytest

from neuromethyl_ont.data import read_simulator
from neuromethyl_ont.data.read_simulator import ReadModel, simulate_read_counts
from neuromethyl_ont.data.sequencing_simulator import encode_batch, simulate_counts

GOLDEN = Path(__file__).parent / "poisson_simulator_golden.npz"


@pytest.fixture
def model(monkeypatch) -> ReadModel:
    """Tiny genome (1.5 Mb) with 40 clusters of 8 probes 150 bp apart, columns shuffled."""
    monkeypatch.setattr(read_simulator, "HG38_CHROM_LENGTHS", {"chr1": 1_000_000, "chr2": 500_000})
    rng = np.random.default_rng(0)
    centers = rng.integers(0, 1_400_000, size=40)
    flat = (centers[:, None] + 150 * np.arange(8)).ravel()
    chrom = np.where(flat < 1_000_000, "chr1", "chr2")
    pos = np.where(flat < 1_000_000, flat, flat - 1_000_000)
    perm = rng.permutation(len(flat))
    probs, length = np.array([0.0, 1.0]), np.array([1500.0, 1500.0])
    return ReadModel.from_coordinates(chrom[perm], pos[perm], (probs, length))


def test_poisson_mode_is_bit_identical_to_the_stored_golden():
    g = np.load(GOLDEN)
    c = simulate_counts(g["beta"], g["lam"], np.random.default_rng(7))
    for key, got in (("m", c.m), ("u", c.u), ("n", c.n), ("full", c.full_array)):
        np.testing.assert_array_equal(got, g[key])
    for r in ("binary", "continuous", "evidence"):
        x = encode_batch(r, g["beta"], np.random.default_rng(11), mu=g["mu"], kappa=2.0)
        np.testing.assert_array_equal(x, g["batch_" + r])


def test_read_mode_is_deterministic_and_counts_are_consistent(model):
    rng = np.random.default_rng(1)
    beta = rng.random((4, 320))
    beta[0, :5] = np.nan
    lam = np.array([0.5, 2.0, np.inf, 1.0])
    a = simulate_read_counts(beta, lam, np.random.default_rng(5), model)
    b = simulate_counts(beta, lam, np.random.default_rng(5), read_model=model)
    c = simulate_read_counts(beta, lam, np.random.default_rng(6), model)
    np.testing.assert_array_equal(a.n, b.n)
    np.testing.assert_array_equal(a.m, b.m)
    assert not np.array_equal(a.n, c.n)
    for arr in (a.m, a.u, a.n):
        assert arr.dtype == np.int64 and arr.min() >= 0
    np.testing.assert_array_equal(a.m + a.u, a.n)
    assert a.full_array.tolist() == [False, False, True, False]
    assert a.n[2].sum() == 0  # full-array rows carry no sampled evidence
    assert a.n[0, :5].sum() == 0  # missing beta = no evidence
    assert a.n[[0, 1, 3]].sum() > 0


def test_counts_are_drawn_from_beta_not_scaled(model):
    ones = simulate_read_counts(np.ones((1, 320)), 4.0, np.random.default_rng(2), model)
    zeros = simulate_read_counts(np.zeros((1, 320)), 4.0, np.random.default_rng(2), model)
    np.testing.assert_array_equal(ones.m, ones.n)
    assert zeros.m.sum() == 0 and zeros.n.sum() > 0
    half = simulate_read_counts(np.full((1, 320), 0.5), 40.0, np.random.default_rng(2), model)
    assert 0.45 < half.m.sum() / half.n.sum() < 0.55


def test_one_read_covers_a_contiguous_cluster_of_probes(model):
    rng = np.random.default_rng(3)
    sizes = []
    for _ in range(300):
        lo, hi = model.read_spans(1, rng)
        covered = np.flatnonzero(model.coverage(lo, hi)[model.feature_idx] > 0)
        np.testing.assert_array_equal(covered, np.arange(lo[0], hi[0]))
        sizes.append(len(covered))
    assert max(sizes) <= 8 + 1 and max(sizes) > 1  # a 1.5 kb read spans <= 1 cluster of 150 bp spacing


def test_expected_depth_matches_lambda(model):
    beta = np.full((1, 320), 0.5)
    mean_n = np.mean(
        [simulate_read_counts(beta, 2.0, np.random.default_rng(s), model).n.mean() for s in range(20)]
    )
    assert 1.7 < mean_n < 2.3


def test_probes_off_the_primary_assembly_are_never_covered(tmp_path, monkeypatch):
    monkeypatch.setattr(read_simulator, "HG38_CHROM_LENGTHS", {"chr1": 10_000})
    manifest = tmp_path / "man.tsv"
    manifest.write_text(
        "Probe_ID\tCpG_chrm\tCpG_beg\ncgA\tchr1\t1000\ncgB\tchr1_KI270706v1_random\t5\n"
        "cgC\tchr1\t2000\n"
    )
    m = ReadModel.from_manifest(["cgA", "cgB", "cgMissing", "cgC"], manifest)  # real length table
    assert m.feature_idx.tolist() == [0, 3] and m.n_features == 4
    n = simulate_read_counts(np.full((1, 4), 0.5), 5.0, np.random.default_rng(0), m).n[0]
    assert n[1] == 0 and n[2] == 0 and n[0] > 0 and n[3] > 0


def test_read_mode_is_selectable_in_encode_batch_depth_curve_and_trainer_config(model):
    from neuromethyl_ont.evaluation.depth_curve import build_fixed_depth_realizations
    from neuromethyl_ont.training.trainer import TrainingConfig, _encode

    beta = np.random.default_rng(4).random((6, 320))
    mu = beta.mean(axis=0)
    for rep in ("binary", "continuous", "evidence"):
        a = encode_batch(rep, beta, np.random.default_rng(9), mu=mu, kappa=2.0, read_model=model)
        b = encode_batch(rep, beta, np.random.default_rng(9), mu=mu, kappa=2.0, read_model=model)
        assert a.shape == beta.shape and np.isfinite(a).all()
        np.testing.assert_array_equal(a, b)
    cfg = TrainingConfig(representation="evidence", read_model=model)
    assert _encode(cfg, beta, mu, np.random.default_rng(9)).shape == beta.shape
    real = build_fixed_depth_realizations(beta, (1.0, np.inf), read_model=model)
    assert real[1.0].n.sum() > 0 and real[np.inf].full_array.all()
