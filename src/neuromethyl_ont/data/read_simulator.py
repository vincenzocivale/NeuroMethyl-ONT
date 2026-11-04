"""Read-structured depth simulator (second mode next to the i.i.d. Poisson one).

The Poisson mode of ``sequencing_simulator.simulate_counts`` draws ``n_i`` independently per
probe. A real nanopore read spans several kb and covers a *cluster* of neighbouring probes, so
low-coverage real samples carry less information than that assumes. Here we simulate reads:

    start ~ uniform over the hg38 primary chromosomes (weighted by length)
    length ~ empirical MethyLYZR ONT_R10 read-length distribution (quantile table, see below)
    every probe whose hg38 coordinate falls inside [start, start + length) is covered
    one binary call per (read, probe) ~ Bernoulli(beta_i)

Summing independent Bernoulli(beta_i) calls over the n_i reads covering probe i is exactly
``m_i ~ Binomial(n_i, beta_i)``, so ``n_i`` comes from the read geometry and ``(m_i, u_i)`` are
always drawn from beta (never ``beta * coverage``). Strand is not drawn: a read on either strand
covers the same CpG positions, so it cannot change any count.

The saved table holds lengths of reads that cover at least one index probe, which over-represents
long reads (a read of length L is seen with probability P(L)). It is converted to the all-reads
distribution by weighting with 1 / P(L), where P(L) is computed exactly from the probe positions.
Simulated reads that cover at least one probe then have the table's length distribution again.

Depth keeps the Poisson convention for the interface: ``lam`` is the expected number of reads per
probe, converted to a number of reads with the geometry's mean probes-per-read.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from neuromethyl_ont.data.sequencing_simulator import SimulatedCounts

# GRCh38 primary chromosome lengths (GCA_000001405.15 no_alt_analysis_set .fai).
HG38_CHROM_LENGTHS: dict[str, int] = {
    "chr1": 248956422, "chr2": 242193529, "chr3": 198295559, "chr4": 190214555,
    "chr5": 181538259, "chr6": 170805979, "chr7": 159345973, "chr8": 145138636,
    "chr9": 138394717, "chr10": 133797422, "chr11": 135086622, "chr12": 133275309,
    "chr13": 114364328, "chr14": 107043718, "chr15": 101991189, "chr16": 90338345,
    "chr17": 83257441, "chr18": 80373285, "chr19": 58617616, "chr20": 64444167,
    "chr21": 46709983, "chr22": 50818468, "chrX": 156040895, "chrY": 57227415,
}  # fmt: skip

READ_LENGTH_TABLE = (
    Path(__file__).resolve().parents[3] / "configs" / "data" / "ont_r10_read_length_quantiles.json"
)
_PILOT_READS = 50_000
_SUPPORT = 2000  # equal-mass points of the length table used as the sampling support


def load_read_length_table(path: str | Path = READ_LENGTH_TABLE) -> tuple[np.ndarray, np.ndarray]:
    """``(probs, length_bp)`` of the saved empirical read-length quantile table."""
    table = json.loads(Path(path).read_text())
    return np.asarray(table["probs"], dtype=np.float64), np.asarray(table["length_bp"], np.float64)


@dataclass(frozen=True)
class ReadModel:
    """Probe positions on the concatenated hg38 primary chromosomes + read-length distribution.

    ``gpos`` are the sorted global coordinates of the probes that sit on a primary chromosome;
    ``feature_idx[j]`` is the column of the beta matrix (feature-index order) of the j-th one.
    Probes without a primary-chromosome coordinate are never covered (n = 0, missing evidence).
    """

    gpos: np.ndarray
    feature_idx: np.ndarray
    n_features: int
    chrom_end: np.ndarray
    support: np.ndarray
    cdf: np.ndarray
    mean_probes_per_read: float

    @classmethod
    def from_coordinates(
        cls,
        chrom: np.ndarray,
        pos: np.ndarray,
        length_table: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> ReadModel:
        """``chrom``/``pos``: hg38 chromosome name and 0-based coordinate per feature column."""
        probs, length_bp = length_table if length_table is not None else load_read_length_table()
        sizes = np.asarray(list(HG38_CHROM_LENGTHS.values()), dtype=np.int64)
        ends = np.cumsum(sizes)
        offset = dict(zip(HG38_CHROM_LENGTHS, ends - sizes, strict=True))
        chrom = np.asarray(chrom, dtype=object)
        keep = np.array([c in offset for c in chrom])
        feature_idx = np.flatnonzero(keep)
        gpos_all = np.array([offset[c] for c in chrom[keep]], dtype=np.int64) + np.asarray(pos)[keep]
        order = np.argsort(gpos_all, kind="stable")
        gpos = gpos_all[order]
        # Table (reads with >= 1 probe) -> all reads: weight by 1 / P(L), P(L) = share of read
        # starts whose read reaches a probe = sum_j min(gap_j, L) / genome length.
        support = np.interp((np.arange(_SUPPORT) + 0.5) / _SUPPORT, probs, length_bp)
        gaps = np.sort(np.diff(gpos, prepend=0)).astype(np.float64)
        prefix = np.concatenate([[0.0], np.cumsum(gaps)])
        below = np.searchsorted(gaps, support, "right")
        p_len = (prefix[below] + support * (len(gaps) - below)) / ends[-1]
        cdf = np.cumsum(1.0 / p_len)
        model = cls(
            gpos=gpos,
            feature_idx=feature_idx[order],
            n_features=len(chrom),
            chrom_end=ends,
            support=support.astype(np.int64),
            cdf=cdf / cdf[-1],
            mean_probes_per_read=1.0,
        )
        lo, hi = model.read_spans(_PILOT_READS, np.random.default_rng(0))
        if not (hi > lo).any():
            raise ValueError("no pilot read covers a probe: check coordinates and chromosome names")
        return replace(model, mean_probes_per_read=float((hi - lo).mean()))

    @classmethod
    def from_manifest(cls, probe_ids: list[str], manifest_path: str | Path) -> ReadModel:
        """Positions from the HM450 hg38 manifest (``Probe_ID``, ``CpG_chrm``, ``CpG_beg``)."""
        import pandas as pd

        man = pd.read_csv(
            manifest_path, sep="\t", usecols=["Probe_ID", "CpG_chrm", "CpG_beg"]
        ).set_index("Probe_ID")
        man = man.reindex(probe_ids)  # probes absent from the manifest get NaN -> never covered
        return cls.from_coordinates(
            man["CpG_chrm"].fillna("NA").to_numpy(), man["CpG_beg"].fillna(0).to_numpy(np.int64)
        )

    def read_spans(self, n_reads: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Covered probe range ``[lo, hi)`` (in sorted-probe order) of each of ``n_reads`` reads."""
        total = int(self.chrom_end[-1])
        start = rng.integers(0, total, size=n_reads)
        length = self.support[np.searchsorted(self.cdf, rng.random(n_reads))]
        end = np.minimum(start + length, self.chrom_end[np.searchsorted(self.chrom_end, start, "right")])
        return np.searchsorted(self.gpos, start, "left"), np.searchsorted(self.gpos, end, "left")

    def coverage(self, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        """Number of reads covering each feature column, from spans (``(n_features,)`` int64)."""
        size = len(self.gpos) + 1
        cover = np.cumsum(np.bincount(lo, minlength=size) - np.bincount(hi, minlength=size))
        n = np.zeros(self.n_features, dtype=np.int64)
        n[self.feature_idx] = cover[:-1]
        return n

    def n_reads_for_depth(self, lam: float) -> int:
        """Reads giving ``lam`` expected reads per probe (the Poisson mode's convention)."""
        return int(round(lam * len(self.gpos) / self.mean_probes_per_read))


def simulate_read_counts(
    beta: np.ndarray,
    lam: np.ndarray | float,
    rng: np.random.Generator,
    read_model: ReadModel,
) -> SimulatedCounts:
    """Read-structured counterpart of ``simulate_counts`` (same arguments and return type).

    ``lam`` as in the Poisson mode (expected reads per probe; ``inf`` = full array, closed form
    downstream). Each sample gets its own independent set of reads.
    """
    beta = np.asarray(beta, dtype=np.float64)
    lam_arr = np.asarray(lam, dtype=np.float64)
    if lam_arr.ndim == 0:
        lam_arr = np.full(beta.shape[0], float(lam_arr))
    if lam_arr.shape[0] != beta.shape[0]:
        raise ValueError("lam must be a scalar or have one entry per sample (beta.shape[0])")
    if beta.shape[1] != read_model.n_features:
        raise ValueError("beta columns must match the read model's feature index")

    full_array = np.isinf(lam_arr)
    missing = np.isnan(beta)
    p = np.clip(np.where(missing, 0.0, beta), 0.0, 1.0)
    n = np.zeros(beta.shape, dtype=np.int64)
    for i in np.flatnonzero(~full_array):
        n[i] = read_model.coverage(*read_model.read_spans(read_model.n_reads_for_depth(lam_arr[i]), rng))
    n[missing] = 0
    m = rng.binomial(n, p).astype(np.int64)
    return SimulatedCounts(m=m, u=n - m, n=n, full_array=full_array)
