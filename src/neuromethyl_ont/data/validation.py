"""Invariant checks for canonical methylation representations.

Each ``validate_*`` function returns a list of human-readable violation
strings (empty list = valid). They never mutate input and never raise on
data problems themselves -- callers decide whether to log, fail the sample,
or raise via :func:`assert_valid`.
"""

from __future__ import annotations

import pandas as pd

DEFAULT_TOLERANCE = 1e-6


class ValidationError(ValueError):
    """Raised by :func:`assert_valid` when violations are present."""


def assert_valid(violations: list[str], context: str = "") -> None:
    if violations:
        prefix = f"{context}: " if context else ""
        raise ValidationError(prefix + "; ".join(violations))


def validate_coordinates(df: pd.DataFrame) -> list[str]:
    """chrom non-empty, start >= 0, end > start."""
    violations: list[str] = []
    if df.empty:
        return violations

    if (df["chrom"].isna() | (df["chrom"].astype(str).str.len() == 0)).any():
        violations.append("empty chrom value present")
    if (df["start"] < 0).any():
        violations.append("negative start coordinate present")
    if (df["end"] <= df["start"]).any():
        violations.append("end <= start present (end must be > start)")
    return violations


def validate_counts(df: pd.DataFrame) -> list[str]:
    """coverage/m/u >= 0 and methylated_count + unmethylated_count == coverage,
    for every row where these are not missing.
    """
    violations: list[str] = []
    if df.empty:
        return violations

    m = df["methylated_count"]
    u = df["unmethylated_count"]
    cov = df["coverage"]
    present = m.notna() & u.notna() & cov.notna()

    if present.any():
        if (m[present] < 0).any():
            violations.append("negative methylated_count present")
        if (u[present] < 0).any():
            violations.append("negative unmethylated_count present")
        if (cov[present] < 0).any():
            violations.append("negative coverage present")
        mismatch = (m[present] + u[present]) != cov[present]
        if mismatch.any():
            violations.append(
                f"methylated_count + unmethylated_count != coverage in {int(mismatch.sum())} row(s)"
            )

    # m and u are always derived together: either both present (there was a
    # methylation percentage to reconstruct from) or both missing. Coverage
    # may legitimately be present while m/u are missing (coverage measured,
    # methylation percentage NA) -- that is not an inconsistency.
    m_xor_u_missing = m.isna() != u.isna()
    if m_xor_u_missing.any():
        violations.append(
            f"methylated_count and unmethylated_count disagree on missingness in "
            f"{int(m_xor_u_missing.sum())} row(s)"
        )

    counts_present_no_coverage = m.notna() & u.notna() & cov.isna()
    if counts_present_no_coverage.any():
        violations.append(
            f"methylated_count/unmethylated_count present but coverage missing in "
            f"{int(counts_present_no_coverage.sum())} row(s)"
        )
    return violations


def validate_beta_range(df: pd.DataFrame, tolerance: float = DEFAULT_TOLERANCE) -> list[str]:
    """0 <= methylation_fraction <= 1, and it matches methylated_count/coverage
    wherever coverage > 0.

    ``methylation_fraction`` comes from a published percentage rounded to two
    decimals, while ``methylated_count`` is the nearest integer reconstruction
    (``round(coverage * fraction)``, see project brief section 6). At low
    coverage this rounding can shift the ratio by up to ``0.5 / coverage``, so
    the per-row tolerance scales with coverage rather than using a single
    fixed epsilon.
    """
    violations: list[str] = []
    if df.empty:
        return violations

    beta = df["methylation_fraction"]
    present = beta.notna()
    if present.any():
        if ((beta[present] < 0) | (beta[present] > 1)).any():
            violations.append("methylation_fraction outside [0,1]")

    cov = df["coverage"]
    m = df["methylated_count"]
    has_all = present & cov.notna() & m.notna() & (cov > 0)
    if has_all.any():
        cov_f = cov[has_all].astype("float64")
        expected = m[has_all].astype("float64") / cov_f
        actual = beta[has_all].astype("float64")
        row_tolerance = 0.5 / cov_f + tolerance
        bad = (expected - actual).abs() > row_tolerance
        if bad.any():
            violations.append(
                f"methylation_fraction inconsistent with methylated_count/coverage in "
                f"{int(bad.sum())} row(s) beyond rounding tolerance (base tolerance={tolerance})"
            )
    return violations


def validate_probe_aggregation(
    coord_df: pd.DataFrame, probe_df: pd.DataFrame, tolerance: float = DEFAULT_TOLERANCE
) -> list[str]:
    """Total coverage over coordinate-level rows with evidence must equal
    total coverage over probe-level aggregate rows, per sample.
    """
    violations: list[str] = []

    coord_has_evidence = coord_df["methylated_count"].notna() & coord_df["unmethylated_count"].notna()
    coord_cov = (
        coord_df.loc[coord_has_evidence]
        .groupby("sample_id")["coverage"]
        .sum()
        .astype("float64")
    )
    probe_cov = (
        probe_df.dropna(subset=["coverage"])
        .groupby("sample_id")["coverage"]
        .sum()
        .astype("float64")
    )

    all_samples = set(coord_cov.index) | set(probe_cov.index)
    for sample_id in sorted(all_samples):
        a = coord_cov.get(sample_id, 0.0)
        b = probe_cov.get(sample_id, 0.0)
        if abs(a - b) > tolerance:
            violations.append(
                f"sample {sample_id}: coordinate-level coverage sum ({a}) != "
                f"probe-level coverage sum ({b})"
            )

    # Aggregated per-probe m+u must equal coverage (redundant with
    # validate_counts, but checked here specifically post-aggregation).
    violations.extend(
        f"probe-level {v}" for v in validate_counts(probe_df)
    )
    return violations
