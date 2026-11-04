# Read-structured depth simulator (R13)

Second mode of `neuromethyl_ont.data.sequencing_simulator.simulate_counts`, next to the i.i.d.
Poisson mode of `docs/EVIDENCE_LINEAR_V1.md` section 7. The Poisson mode stays the default and is
unchanged (a stored golden fixture, `tests/data/poisson_simulator_golden.npz`, pins it bit for bit).

## What it simulates

Code: `src/neuromethyl_ont/data/read_simulator.py`.

- N reads. Start uniform over the 24 hg38 primary chromosomes (so weighted by chromosome length),
  length from the empirical ONT R10 distribution, clipped at the end of the chromosome.
- A probe is covered by a read if its hg38 coordinate lies in `[start, start + length)`. A read
  therefore covers a genomically contiguous cluster of probes.
- One binary call per (read, probe), `~ Bernoulli(beta_i)`. Summing the calls of the `n_i` reads on
  probe `i` is exactly `m_i ~ Binomial(n_i, beta_i)`, so `(m, u)` are always drawn from beta (never
  `beta * coverage`) and `n_i` comes from the read geometry. Missing beta gives `n = m = u = 0`.
- Strand is not drawn: a read on either strand covers the same CpG positions, so it cannot change a
  count.
- Interface: same arguments and return type (`SimulatedCounts`). `lam` keeps the Poisson meaning
  "expected reads per probe" and is converted with the geometry's mean probes per read
  (`ReadModel.n_reads_for_depth`), so `LAMBDA_GRID`, `DEPTH_GRID` and the training mixture work in
  both modes at matched expected coverage. `lam = inf` is still the full array.

Select the mode by passing a `ReadModel`; `None` (default) is the old behaviour:

```python
from neuromethyl_ont.data.read_simulator import ReadModel
model = ReadModel.from_manifest(probe_ids, f"{root}/references/hg38/HM450.hg38.manifest.tsv.gz")
simulate_counts(beta, lam, rng, read_model=model)
encode_batch(rep, beta, rng, mu=mu, kappa=k, read_model=model)
build_fixed_depth_realizations(beta, read_model=model)   # and run_depth_curve(..., read_model=model)
TrainingConfig(representation="evidence", read_model=model)
```

Not wired into any training/evaluation script yet (only the library entry points above).

## Probe coordinates

`$DATA_ROOT/references/hg38/HM450.hg38.manifest.tsv.gz` (`Probe_ID`, `CpG_chrm`, `CpG_beg`, 0-based
start of the CpG). All 428,799 feature-index probes are in it; 428,697 sit on chr1-22/X/Y and get a
position, 102 are on alt/random/unplaced contigs or have no coordinate and are never covered
(`n = 0`, no evidence). The 24 chromosome lengths are a constant in the module (GRCh38 no_alt
analysis set `.fai`).

## Read lengths

`configs/data/ont_r10_read_length_quantiles.json`: quantile table (27 points, top 0.1% clamped to
the 99.9th percentile, the raw maximum is a 348 kb outlier) of per-read `read_length` pooled over 10
of the 207 MethyLYZR ONT_R10 feather files (1.47 M reads; observed median 4.8 kb, mean 6.5 kb).
**This is R10 PromethION data** (MethyLYZR, Zenodo 13236097), not MinION as in Sturgeon V1. Files
were extracted from the archive into a scratch directory, read with `python -I`, and deleted; only
the table is stored. Rebuild: `scripts/data/build_read_length_table.py`.

The files hold read x probe rows, so a read appears only if it covers at least one index probe: the
table is length-biased towards long reads. `ReadModel` converts it to the all-reads distribution by
weighting with `1 / P(L)`, `P(L)` = share of read starts whose read reaches a probe, computed exactly
from the probe positions. After the conversion the mean length drops from 6.5 kb to 3.3 kb (median 1.7 kb) and
only about 13% of simulated reads cover any probe (mean 0.47 probes per read over all reads, about
3.5 per informative read). Not converting double-counts the bias (4.3 probes per informative read
against 3.2 in the real files).

## Validation against real reads

`scripts/evaluation/validate_read_simulator.py`, table in `docs/READ_SIMULATOR_VALIDATION.md`.
Real curve: 5 further R10 files (not in the length table), reads shuffled with a fixed seed,
cumulative distinct probes against N reads. All 427,950 distinct `epic_id` seen in 15 real files are
in our 428,799-probe index (the MethyLYZR files are already limited to 450K-compatible probes);
they cover 99.8% of the index.

Findings (N = reads covering at least one probe):

1. At the same expected number of probe calls, the read mode and Poisson give the same marginal
   coverage (covered fraction within 1%, same mean and sd of reads per covered probe), because
   coverage is far below one read per probe and a read's probes are spread over many probes.
   Poisson is therefore **not optimistic in the per-probe count distribution**.
2. Both are above the real data in covered fraction: +5-9% from 300 to 10k reads, +12% at 30k,
   +9% at 100k reads (k = 3 files). The real reads carry slightly fewer probes (3.2 against 3.5 per read) and
   are more unevenly spread (sd of reads per covered probe 0.71 against 0.37 at 30k reads), likely
   copy-number and mappability effects that uniform starts do not model.
3. What differs is the joint structure: a read gives about 3.5 probe calls in one genomic cluster
   (covered probes per run of adjacent covered probes: 3.4-7.0 simulated, 2.8-5.5 real, 1.0-2.2
   Poisson). The read mode reproduces this clustering; i.i.d. Poisson has none, so at equal numbers
   of calls it overstates how many independent loci a sample touches.

## Limits

- Within a read the calls are independent given beta (real reads share one molecule's methylation
  state), so the read mode captures the clustering of coverage, not correlated calls.
- Cost: reads that cover no probe are 87% of the draws. N = lam x 428,799 / 0.47 reads per sample,
  measured at about 0.5 s per sample at lam = 1 (0.9 M reads) and 4 s at lam = 8, against 0.05 s
  for Poisson; lam = 32 is extrapolated to about 16 s. Fine for the fixed depth curve, slow for
  training at high lam (a thinned informative-reads-only sampler would fix it; not built).
- Uniform starts over the whole assembly (including N gaps), no mappability, GC or CNV bias, no
  mapped-length vs raw-length difference, one length table pooled over 10 libraries.
