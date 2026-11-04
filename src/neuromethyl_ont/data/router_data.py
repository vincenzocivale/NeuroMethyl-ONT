"""Feature construction for the CNS-vs-leukemia domain router.

Every cohort is mapped onto the GSE90496 feature index, binarized like Binary
(+1/-1 around the 0.6 threshold, 0 = unobserved) and restricted to probes also
present in the ALMA reference (everything else is set to 0).
"""

from __future__ import annotations

import numpy as np

from neuromethyl_ont.data.array_reference import FeatureIndex
from neuromethyl_ont.data.sequencing_simulator import BINARY_THRESHOLD


def shared_probe_mask(feature_index: FeatureIndex, other_probe_ids: list[str]) -> np.ndarray:
    """Boolean mask over ``feature_index`` probes that also exist in ``other_probe_ids``."""
    other = set(other_probe_ids)
    return np.array([p in other for p in feature_index.probe_ids], dtype=bool)


def binary_on_index(
    beta,
    probe_ids: list[str],
    feature_index: FeatureIndex,
    mask: np.ndarray,
    chunk: int = 256,
) -> np.ndarray:
    """(samples x probes) beta -> float32 binary features on ``feature_index``.

    ``beta`` may be a memmap; rows are processed in chunks so the aligned float64
    copy never has to fit in memory.
    """
    column = {p: i for i, p in enumerate(probe_ids)}
    src = np.array([column.get(p, -1) for p in feature_index.probe_ids])
    usable = (src >= 0) & mask
    out = np.zeros((beta.shape[0], feature_index.n_features), dtype=np.float32)
    for start in range(0, beta.shape[0], chunk):
        block = np.asarray(beta[start:start + chunk], dtype=np.float64)
        sub = block[:, src[usable]]
        x = np.where(sub > BINARY_THRESHOLD, 1.0, -1.0)
        x[np.isnan(sub)] = 0.0
        out[start:start + chunk, usable] = x
    return out


def counts_to_binary(binary_features: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply the shared-probe mask to already-binarized ONT features (cohort_scoring)."""
    return (binary_features * mask[None, :]).astype(np.float32)
