"""Load the {Meth, Unmeth, Beta, DetectionP} matrices written by
``scripts/data/idat_to_beta_mnp.R`` (the MNPpreprocessIllumina IDAT importer;
see docs/GSE109379_IDAT_RECONSTRUCTION.md) and compare a reconstruction
against GEO's own (possibly malformed) processed matrix as a correctness
check.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MnpPreprocessedMatrices:
    """One reconstruction run: (n_samples, n_probes) arrays, aligned by
    ``sample_ids``/``probe_ids`` order.
    """

    sample_ids: list[str]
    probe_ids: list[str]
    meth: np.ndarray
    unmeth: np.ndarray
    beta: np.ndarray
    detection_p: np.ndarray


def load_mnp_preprocessed(output_dir: str | Path) -> MnpPreprocessedMatrices:
    """Read the binary output of ``idat_to_beta_mnp.R``.

    Each ``*.f8`` file is a column-major flatten of an (n_probes x n_samples)
    R matrix, which is byte-for-byte a row-major flatten of its
    (n_samples x n_probes) transpose -- so a plain
    ``np.fromfile(..., dtype="<f8").reshape(n_samples, n_probes)`` recovers
    the matrix in this repo's usual (samples, probes) orientation.
    """
    output_dir = Path(output_dir)
    probe_ids = (output_dir / "probe_ids.txt").read_text().splitlines()
    sample_ids = (output_dir / "sample_ids.txt").read_text().splitlines()
    n_probes, n_samples = len(probe_ids), len(sample_ids)

    def _load(name: str) -> np.ndarray:
        arr = np.fromfile(output_dir / name, dtype="<f8")
        expected = n_probes * n_samples
        if arr.size != expected:
            raise ValueError(
                f"{output_dir / name}: expected {expected} values "
                f"({n_samples} samples x {n_probes} probes), got {arr.size}"
            )
        return arr.reshape(n_samples, n_probes)

    return MnpPreprocessedMatrices(
        sample_ids=sample_ids,
        probe_ids=probe_ids,
        meth=_load("meth.f8"),
        unmeth=_load("unmeth.f8"),
        beta=_load("beta.f8"),
        detection_p=_load("detection_p.f8"),
    )


def compare_beta_matrices(
    reconstructed_beta: np.ndarray,
    reconstructed_probe_ids: list[str],
    reference_beta: np.ndarray,
    reference_probe_ids: list[str],
) -> dict:
    """Compare two (n_samples, n_probes) beta matrices sharing the same
    sample order, over the intersection of their probe ids.

    Returns a dict with ``n_common_probes``, ``pearson_r`` (flattened, over
    all sample x probe pairs), ``median_abs_diff``, ``max_abs_diff``, and
    ``fraction_within_0_05``. Raises if the two matrices don't have the same
    number of samples (a caller bug, not a data-quality finding).
    """
    if reconstructed_beta.shape[0] != reference_beta.shape[0]:
        raise ValueError(
            f"sample count mismatch: reconstructed has {reconstructed_beta.shape[0]}, "
            f"reference has {reference_beta.shape[0]} -- align sample order before comparing"
        )

    common = pd.Index(reconstructed_probe_ids).intersection(reference_probe_ids)
    if len(common) == 0:
        raise ValueError("no probe ids in common between the two matrices")

    recon_pos = pd.Index(reconstructed_probe_ids).get_indexer(common)
    ref_pos = pd.Index(reference_probe_ids).get_indexer(common)
    a = reconstructed_beta[:, recon_pos].ravel().astype(np.float64)
    b = reference_beta[:, ref_pos].ravel().astype(np.float64)

    valid = ~(np.isnan(a) | np.isnan(b))
    a, b = a[valid], b[valid]
    diff = np.abs(a - b)

    return {
        "n_common_probes": int(len(common)),
        "n_compared_values": int(valid.sum()),
        "pearson_r": float(np.corrcoef(a, b)[0, 1]) if len(a) > 1 else float("nan"),
        "median_abs_diff": float(np.median(diff)) if len(diff) else float("nan"),
        "max_abs_diff": float(np.max(diff)) if len(diff) else float("nan"),
        "fraction_within_0_05": float((diff <= 0.05).mean()) if len(diff) else float("nan"),
    }
