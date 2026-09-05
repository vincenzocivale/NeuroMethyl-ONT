"""Mandatory tests from docs/EVIDENCE_LINEAR_V1.md #20."""

import math

import numpy as np

from neuromethyl_ont.data.sequencing_simulator import (
    LAMBDA_GRID,
    LAMBDA_WEIGHTS,
    binary_transform,
    continuous_transform,
    evidence_transform,
    simulate_counts,
)


def test_evidence_transform_reference_value():
    # mu=0.5, m=8, u=2, n=10, kappa=2 -> posterior=(8+2*0.5)/12=0.75 -> x=0.25
    x = evidence_transform(m=np.array([8.0]), u=np.array([2.0]), mu=np.array([0.5]), kappa=2.0)
    assert math.isclose(x[0], 0.25, abs_tol=1e-9)


def test_missing_n_zero_gives_zero_not_nan():
    m = np.array([0.0])
    u = np.array([0.0])
    mu = np.array([0.5])

    x_evidence = evidence_transform(m, u, mu, kappa=2.0)
    x_continuous = continuous_transform(m, u, mu)
    x_binary = binary_transform(m, u)

    for x in (x_evidence, x_continuous, x_binary):
        assert not np.isnan(x[0])
        assert x[0] == 0.0

    # kappa=0 with n=0 is the indeterminate 0/0 case before masking.
    x_evidence_kappa0 = evidence_transform(m, u, mu, kappa=0.0)
    assert not np.isnan(x_evidence_kappa0[0])
    assert x_evidence_kappa0[0] == 0.0


def test_high_coverage_limit_matches_beta_hat_minus_mu():
    mu = np.array([0.5])
    beta_hat = 0.9
    n = 1_000_000.0
    m = np.array([beta_hat * n])
    u = np.array([n - beta_hat * n])

    x = evidence_transform(m, u, mu, kappa=2.0)
    assert math.isclose(x[0], beta_hat - mu[0], abs_tol=1e-5)


def test_coverage_effect_low_n_shrinks_more_than_high_n():
    mu = np.array([0.5])
    beta_hat = 0.9
    kappa = 2.0

    x_n1 = evidence_transform(np.array([0.9 * 1]), np.array([0.1 * 1]), mu, kappa)
    x_n20 = evidence_transform(np.array([0.9 * 20]), np.array([0.1 * 20]), mu, kappa)

    asymptote = beta_hat - mu[0]
    assert 0 < x_n1[0] < x_n20[0] < asymptote


def test_lambda_mixture_sums_to_one_and_includes_full_array_sentinel():
    assert math.isclose(LAMBDA_WEIGHTS.sum(), 1.0)
    assert np.isinf(LAMBDA_GRID[-1])


def test_simulator_counts_are_consistent_and_unbiased():
    rng = np.random.default_rng(0)
    beta = np.full((1, 20_000), 0.3)
    counts = simulate_counts(beta, lam=50.0, rng=rng)

    assert (counts.m >= 0).all()
    assert (counts.u >= 0).all()
    assert (counts.m + counts.u == counts.n).all()
    assert not counts.full_array.any()

    empirical_beta = counts.m.sum() / counts.n.sum()
    assert math.isclose(empirical_beta, 0.3, abs_tol=0.02)


def test_simulator_missing_probes_stay_missing_at_every_depth():
    rng = np.random.default_rng(0)
    beta = np.array([[0.4, np.nan, 0.9]])
    counts = simulate_counts(beta, lam=32.0, rng=rng)

    assert counts.n[0, 1] == 0
    assert counts.m[0, 1] == 0
    assert counts.u[0, 1] == 0


def test_simulator_full_array_sentinel_flags_the_row():
    rng = np.random.default_rng(0)
    beta = np.array([[0.4, 0.6], [0.4, 0.6]])
    counts = simulate_counts(beta, lam=np.array([32.0, np.inf]), rng=rng)

    assert not counts.full_array[0]
    assert counts.full_array[1]
