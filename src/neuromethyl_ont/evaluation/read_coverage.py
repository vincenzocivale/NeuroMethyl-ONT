"""Coverage statistics of simulated reads vs real MethyLYZR ONT reads (R13 validation).

Real files are per-(read, probe) feather tables (``read_id``, ``epic_id``, ``read_length``).
Everything here treats them as untrusted data: only columns are read, nothing is executed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from neuromethyl_ont.data.read_simulator import ReadModel

QUANTILE_PROBS = np.array(
    [0, 0.005, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6,
     0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99, 0.999]
)  # fmt: skip
# Top 0.1% of reads are clamped to the 99.9th percentile (the raw maximum is a 348 kb outlier).


def read_length_table(files: list[Path], probe_ids: list[str], source: str) -> dict:
    """Pooled per-read length quantiles over the feather files.

    One length per ``read_id``, only reads with at least one row whose ``epic_id`` is in
    ``probe_ids`` (the reads that are informative for our feature index).
    """
    keep = set(probe_ids)
    lengths = []
    for f in files:
        d = pd.read_feather(f, columns=["read_id", "epic_id", "read_length"])
        lengths.append(d[d["epic_id"].isin(keep)].drop_duplicates("read_id")["read_length"])
    all_len = pd.concat(lengths).to_numpy(np.float64)
    return {
        "source": source,
        "note": (
            "Lengths of reads with at least one probe of the 428,799-probe feature index (the "
            "files hold read x probe rows), so long reads are over-represented; ReadModel "
            "re-weights by 1/P(L). ONT R10 (PromethION), MethyLYZR Zenodo 13236097."
        ),
        "n_files": len(files),
        "n_reads": int(all_len.size),
        "mean_bp_observed": round(float(all_len.mean()), 1),
        "probs": QUANTILE_PROBS.tolist(),
        "length_bp": np.quantile(all_len, QUANTILE_PROBS).round(0).tolist(),
    }


def write_json(table: dict, path: Path) -> None:
    path.write_text(json.dumps(table, indent=1) + "\n")


def _stats(n: np.ndarray, order: np.ndarray, n_reads: int) -> dict:
    """Coverage of a per-probe read-count vector ``n`` (feature order).

    ``frac``: covered fraction of all probes; ``mean``/``sd``: reads per covered probe;
    ``per_read``: probe calls per read; ``run``: covered probes per run of genomically adjacent
    covered probes (``order`` = feature columns in genomic order), i.e. the cluster size.
    """
    cov = n[n > 0]
    c = (n[order] > 0).astype(np.int8)
    runs = int((np.diff(c, prepend=0) == 1).sum())
    return {
        "frac": cov.size / n.size,
        "mean": float(cov.mean()),
        "sd": float(cov.std()),
        "per_read": float(n.sum() / n_reads),
        "run": float(c.sum() / runs),
    }


def simulated_read_stats(
    model: ReadModel, n_informative: int, rng: np.random.Generator
) -> tuple[dict, dict, int]:
    """Read mode and Poisson mode at the same expected number of probe observations.

    Reads are simulated until ``n_informative`` of them cover at least one probe (the real files
    only contain such reads). Poisson uses lambda = observations / n_probes. Returns (read
    stats, Poisson stats, reads simulated in total).
    """
    spans: list[tuple[np.ndarray, np.ndarray]] = []
    have = total = 0
    while have < n_informative:
        chunk = max(1000, int((n_informative - have) * 1.3))
        lo, hi = model.read_spans(chunk, rng)
        keep = hi > lo
        spans.append((lo[keep], hi[keep]))
        have += int(keep.sum())
        total += chunk
    lo = np.concatenate([s[0] for s in spans])[:n_informative]
    hi = np.concatenate([s[1] for s in spans])[:n_informative]
    n = model.coverage(lo, hi)
    pois = rng.poisson(n.sum() / n.size, size=n.size)
    return (
        _stats(n, model.feature_idx, n_informative),
        _stats(pois, model.feature_idx, n_informative),
        total,
    )


def real_curve(
    files: list[Path], model: ReadModel, probe_ids: list[str], grid: list[int], seed: int
) -> dict[int, list[dict]]:
    """Per file: shuffle the reads (fixed seed) and report the stats after the first N reads.

    Only rows whose ``epic_id`` is in ``probe_ids`` count; only reads with at least one such
    row are reads. Files with fewer than N reads do not contribute to N.
    """
    index = {p: i for i, p in enumerate(probe_ids)}
    out: dict[int, list[dict]] = {n: [] for n in grid}
    for k, f in enumerate(files):
        d = pd.read_feather(f, columns=["read_id", "epic_id"])
        d["probe"] = d["epic_id"].map(index)
        d = d.dropna(subset=["probe"])
        read_codes, reads = pd.factorize(d["read_id"])
        rank = np.random.default_rng((seed, k)).permutation(len(reads))[read_codes]
        probe = d["probe"].to_numpy(np.int64)
        for n_reads in grid:
            if n_reads <= len(reads):
                n = np.bincount(probe[rank < n_reads], minlength=len(probe_ids))
                out[n_reads].append(_stats(n, model.feature_idx, n_reads))
    return out


def seen_epic_ids(files: list[Path]) -> set[str]:
    return set().union(*(pd.read_feather(f, columns=["epic_id"])["epic_id"].unique() for f in files))
