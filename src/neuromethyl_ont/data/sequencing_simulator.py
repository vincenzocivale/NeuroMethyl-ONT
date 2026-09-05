"""Synthetic sequencing evidence and the B0/C0/E1 representation transforms.

See ``docs/EVIDENCE_LINEAR_V1.md`` for the full spec. GSE90496 has no real
coverage, so training/validation depth is simulated by treating the array
beta value as a latent methylation probability and drawing counts:

    n_i ~ Poisson(lambda)
    m_i ~ Binomial(n_i, beta_i)
    u_i = n_i - m_i

``lambda = np.inf`` is the sentinel for "full array, no sampling" (section 7):
in that limit C0 and E1 both reduce to ``beta_i - mu_i`` and B0 reduces to
thresholding the raw beta directly, rather than counts drawn from it.

The transforms are pure, vectorized numpy functions so they can be reused
identically for training-time augmentation (random lambda per sample) and for
the fixed, saved depth-curve realizations used at evaluation time (same
``(m, u, n)`` draw shared across B0/C0/E1 for a fair comparison).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Section 7 depth mixture: sparse / low-full / high / clean-array.
# Order matters only for readability; sampling uses LAMBDA_GRID/LAMBDA_WEIGHTS together.
LAMBDA_GRID: np.ndarray = np.array(
    [0.02, 0.05, 0.10, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, np.inf]
)
LAMBDA_WEIGHTS: np.ndarray = np.array(
    [
        0.05, 0.05, 0.05,          # sparse: 15%
        0.1125, 0.1125, 0.1125, 0.1125,  # low/full: 45%
        0.10, 0.10, 0.10,          # high: 30%
        0.10,                      # full array (inf): 10%
    ]
)

BINARY_THRESHOLD = 0.6

if not np.isclose(LAMBDA_WEIGHTS.sum(), 1.0):
    raise AssertionError("LAMBDA_WEIGHTS must sum to 1.0")
if LAMBDA_GRID.shape != LAMBDA_WEIGHTS.shape:
    raise AssertionError("LAMBDA_GRID and LAMBDA_WEIGHTS must be the same shape")


@dataclass(frozen=True)
class SimulatedCounts:
    """Synthetic sequencing evidence for a (samples, probes) beta matrix.

    ``full_array`` is a per-sample boolean flag (shape ``(n_samples,)``): rows
    sampled at ``lambda = inf`` carry meaningless ``m``/``u``/``n`` (never
    sampled) and must be handled via the closed-form ``*_full_array``
    transforms instead of ``m``/``u``.
    """

    m: np.ndarray
    u: np.ndarray
    n: np.ndarray
    full_array: np.ndarray


def sample_lambda_batch(rng: np.random.Generator, size: int) -> np.ndarray:
    """Draw ``size`` depth conditions from the section 7 mixture."""
    idx = rng.choice(len(LAMBDA_GRID), size=size, p=LAMBDA_WEIGHTS)
    return LAMBDA_GRID[idx]


def simulate_counts(
    beta: np.ndarray, lam: np.ndarray | float, rng: np.random.Generator
) -> SimulatedCounts:
    """Draw synthetic sequencing counts for every entry of ``beta``.

    ``lam`` is either a scalar (applied to every sample) or a 1-D array of
    length ``beta.shape[0]`` (one depth condition per sample, broadcast
    across that sample's probes -- a single sequencing run has one depth).
    ``lam == inf`` marks a sample as full-array (see ``SimulatedCounts``).
    Missing probes (``NaN`` in ``beta``) always get ``m = u = n = 0``,
    regardless of the depth condition -- missingness at every depth.
    """
    beta = np.asarray(beta, dtype=np.float64)
    lam_arr = np.asarray(lam, dtype=np.float64)
    if lam_arr.ndim == 0:
        lam_arr = np.full(beta.shape[0], float(lam_arr))
    if lam_arr.shape[0] != beta.shape[0]:
        raise ValueError("lam must be a scalar or have one entry per sample (beta.shape[0])")

    full_array = np.isinf(lam_arr)
    lam_safe = np.where(full_array, 1.0, lam_arr).reshape(-1, 1)  # placeholder for full-array rows

    missing = np.isnan(beta)
    p = np.clip(np.where(missing, 0.0, beta), 0.0, 1.0)

    n = rng.poisson(lam_safe, size=beta.shape).astype(np.int64)
    n = np.where(missing, 0, n)
    m = rng.binomial(n, p).astype(np.int64)
    u = n - m

    return SimulatedCounts(m=m, u=u, n=n, full_array=full_array)


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return numerator / denominator


def binary_transform(
    m: np.ndarray, u: np.ndarray, threshold: float = BINARY_THRESHOLD
) -> np.ndarray:
    """B0 feature from synthetic counts: +1/-1 above/at-or-below threshold, 0 if unobserved."""
    m = np.asarray(m, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    n = m + u
    beta_hat = _safe_divide(m, n)
    x = np.where(beta_hat > threshold, 1.0, -1.0)
    return np.where(n == 0, 0.0, x)


def binary_transform_full_array(
    beta: np.ndarray, threshold: float = BINARY_THRESHOLD
) -> np.ndarray:
    """B0 feature straight from array beta (the ``lambda = inf`` closed form)."""
    beta = np.asarray(beta, dtype=np.float64)
    missing = np.isnan(beta)
    x = np.where(beta > threshold, 1.0, -1.0)
    return np.where(missing, 0.0, x)


def continuous_transform(m: np.ndarray, u: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """C0 feature from synthetic counts: beta_hat - mu, 0 if unobserved."""
    m = np.asarray(m, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    n = m + u
    beta_hat = _safe_divide(m, n)
    x = beta_hat - mu
    return np.where(n == 0, 0.0, x)


def continuous_transform_full_array(beta: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """C0 feature straight from array beta (also E1's ``lambda = inf`` limit)."""
    beta = np.asarray(beta, dtype=np.float64)
    missing = np.isnan(beta)
    x = beta - mu
    return np.where(missing, 0.0, x)


def evidence_transform(m: np.ndarray, u: np.ndarray, mu: np.ndarray, kappa: float) -> np.ndarray:
    """E1 feature: coverage-shrunk posterior beta minus the training prior.

    ``posterior = (m + kappa * mu) / (n + kappa)``, ``x = posterior - mu``,
    equivalently ``x = n / (n + kappa) * (beta_hat - mu)``. ``n = 0 -> x = 0``
    (also covers the ``kappa = 0, n = 0`` case, which is an indeterminate
    ``0/0`` before masking). ``kappa = 0`` degenerates to ``continuous_transform``.
    """
    m = np.asarray(m, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    n = m + u
    posterior = _safe_divide(m + kappa * mu, n + kappa)
    x = posterior - mu
    return np.where(n == 0, 0.0, x)


def apply_representation(
    representation: str,
    beta: np.ndarray,
    counts: SimulatedCounts,
    mu: np.ndarray | None = None,
    kappa: float | None = None,
    threshold: float = BINARY_THRESHOLD,
) -> np.ndarray:
    """Dispatch to the sampled-count transform, falling back to the
    closed-form full-array transform for rows with ``counts.full_array``.
    """
    if representation == "binary":
        x_sampled = binary_transform(counts.m, counts.u, threshold=threshold)
        x_full = binary_transform_full_array(beta, threshold=threshold)
    elif representation == "continuous":
        if mu is None:
            raise ValueError("continuous representation requires mu")
        x_sampled = continuous_transform(counts.m, counts.u, mu)
        x_full = continuous_transform_full_array(beta, mu)
    elif representation == "evidence":
        if mu is None or kappa is None:
            raise ValueError("evidence representation requires mu and kappa")
        x_sampled = evidence_transform(counts.m, counts.u, mu, kappa)
        x_full = continuous_transform_full_array(beta, mu)  # lambda -> inf limit
    else:
        raise ValueError(f"unknown representation: {representation!r}")

    full_mask = counts.full_array[:, None] if counts.full_array.ndim == 1 else counts.full_array
    return np.where(full_mask, x_full, x_sampled)


def encode_full_array(
    representation: str,
    beta: np.ndarray,
    mu: np.ndarray | None = None,
    threshold: float = BINARY_THRESHOLD,
) -> np.ndarray:
    """The ``lambda = inf`` closed form, without going through ``simulate_counts``.

    Used for full-array evaluation (spec #13) where every sample is
    deterministically at "clean array" depth -- e.g. per-epoch validation
    during training, ahead of the separate fixed synthetic depth curve.
    """
    if representation == "binary":
        return binary_transform_full_array(beta, threshold=threshold)
    if representation in ("continuous", "evidence"):
        if mu is None:
            raise ValueError(f"{representation} representation requires mu")
        return continuous_transform_full_array(beta, mu)
    raise ValueError(f"unknown representation: {representation!r}")


def encode_batch(
    representation: str,
    beta: np.ndarray,
    rng: np.random.Generator,
    mu: np.ndarray | None = None,
    kappa: float | None = None,
    threshold: float = BINARY_THRESHOLD,
) -> np.ndarray:
    """Training-time augmentation: draw one depth per sample, apply the
    representation transform. Use for random augmentation only -- evaluation
    should call :func:`simulate_counts` once with a fixed seed and reuse the
    same counts across representations (see docs/EVIDENCE_LINEAR_V1.md #13).
    """
    lam = sample_lambda_batch(rng, size=beta.shape[0])
    counts = simulate_counts(beta, lam, rng)
    return apply_representation(
        representation, beta, counts, mu=mu, kappa=kappa, threshold=threshold
    )
