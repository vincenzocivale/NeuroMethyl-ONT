# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research codebase for CNS tumor methylation classification from Oxford Nanopore
sequencing (as opposed to Illumina array data). Core question (see `docs/EVIDENCE_LINEAR_V1.md`): does keeping continuous
beta values and sequencing-depth evidence (instead of collapsing to binary
methylation calls, as Sturgeon/crossNN do) improve classification. No clinical
use is implied.

**Current question.** Can a classifier give *reliable* CNS tumour calls at low coverage, so that more
samples fit on one flow cell? Reliability is measured as accuracy, calibration and abstention against
sequencing depth, on the same samples and class set as the published competitors (Sturgeon, crossNN,
MethyLYZR, ...). Binary/Beta/Evidence below are the tools for that question.
Model names: Binary (formerly B0), Beta (formerly C0), Evidence (formerly E1) and Consensus (formerly B0+E1); `docs/MODEL_NAMES.md` has the mapping.

## Current status and where to work

Done: Binary/Beta/Evidence and the Consensus ensemble trained on GSE90496; evaluated on Rapid-CNS2, ONT-WGS, GSE289246,
GSE209865, Magibu pools and own MB/ETMR samples; competitor runs (Sturgeon, crossNN, MethyLYZR) in
`outputs/benchmarks/` (gitignored, regenerable with `scripts/competitors/`); a *synthetic* depth curve on the GSE90496 holdout
(simulated counts, `scripts/evaluation/evaluate_depth_curve.py`); ALMA leukemia head and router.

Not done:
- No depth curve on **real downsampled** ONT data (Rapid-CNS2 / GSE289246 reads or time-cumulative BAMs).
- No mapping from Sturgeon's released 87 classes to the 91 used here, so no same-class-set comparison.
- `nanoDx_EGA` is empty (controlled access), so there is no read-level downsampling cohort yet.
- The earlier EB1 (Beta-Binomial Naive Bayes) results in `outputs/benchmarks/EB1_VS_COMPETITORS.md` come from
  code that was never committed (`models/evidence_bayes.py`, `docs/EVIDENCE_BAYES_V1.md`); only the
  checkpoint in `outputs/experiments/eb1_empirical_kappa_seed17` remains, so EB1 is not reproducible from this repo.

Work on branch `mission/<slug>`, never on `main`; run `ruff check .` and `pytest` first.

## Commands

```bash
pip install -e '.[dev]'        # install (also: make install)
pytest                         # run all tests (also: make test)
pytest tests/data/test_rapidcns2_parser.py            # single file
pytest tests/data/test_rapidcns2_parser.py::test_name # single test
ruff check .                   # lint (also: make lint)
python scripts/data/validate_registry.py              # validate registry (also: make registry)
```

Run `ruff check .` and `pytest` before committing (see `CONTRIBUTING.md`).

External data is never in this repo. Most commands that touch real datasets
require:

```bash
export NEUROMETHYL_DATA_ROOT=/path/to/CNSNanoporeData
```

`src/neuromethyl_ont/data/paths.py:get_data_root()` reads this env var and
raises if it's unset (unless called with `required=False`). Tests that need
real external data are expected to skip cleanly when the root isn't set —
follow that pattern rather than hardcoding a path.

## Architecture

Strict pipeline, enforced as a layering convention (`docs/ARCHITECTURE.md`),
not by import boundaries:

```
source data -> data readers/harmonization -> canonical sample representations
            -> models -> training/inference -> evaluation -> experiment artifacts
```

- `src/neuromethyl_ont/data/` — schemas, dataset adapters, preprocessing,
  representation conversion. No training-loop logic here.
- `src/neuromethyl_ont/models/` — model definitions only; models consume
  canonical tensors/records and must not know about GEO/EGA filenames or
  dataset-specific quirks.
- `src/neuromethyl_ont/training/` — optimization, checkpointing, experiment
  execution.
- `src/neuromethyl_ont/evaluation/` — metrics, confidence/abstention
  analysis, coverage-scaling evaluation, competitor adapters.
- `src/neuromethyl_ont/utils/` — small infra helpers only; domain logic
  should not accumulate here.
- `scripts/` — thin CLI entry points (`data/`, `training/`, `inference/`,
  `competitors/`, `evaluation/`, `figures/`). If a script grows real logic, move it into `src/` and test
  it there; scripts should just parse args/config and call package code.

Everything is driven by versioned YAML under `configs/{data,models,experiments}/`
rather than CLI flags for anything experiment-defining.

### Canonical methylation representations

The project supports progressively richer representations of the same
underlying signal, and keeping conversions between them explicit is
important for fair ablations (`docs/ARCHITECTURE.md`):

1. **Binary array-like**: `(probe_id, state)`
2. **Continuous**: `(locus, beta)`
3. **Coverage-aware**: `(locus, methylated_count, unmethylated_count, beta, coverage)`
4. **Nanopore-native**: variable-size locus/read sets with genomic context

The active experiment (`docs/EVIDENCE_LINEAR_V1.md`) implements exactly this
as three linear models sharing one architecture (`nn.Linear(n_features, 91)`)
and one GSE90496-derived feature index, differing only in feature
construction: **Binary** (`BinaryLinear`, threshold at 0.6), **Beta**
(`ContinuousLinear`, beta minus training-set mean), **Evidence**
(`EvidenceLinearV1`, coverage-shrunk posterior beta minus mean via a `kappa`
prior-strength parameter — `kappa=0` degenerates Evidence to Beta). Sequencing depth
must only ever act as *confidence in the evidence* (shrinkage), never as an
independent input feature — otherwise a model could learn
depth→batch→tumor-class shortcuts instead of biology. When touching
`sequencing_simulator.py`, always sample `(m, u) ~ Poisson/Binomial` from
`beta` first and derive features from the counts — never compute
`beta * coverage` directly.

### Rapid-CNS2 / ONT-WGS canonical preprocessing

`docs/RAPIDCNS2_CANONICAL_PREPROCESSING.md` is the source of truth for
`src/neuromethyl_ont/data/{rapidcns2.py,canonical.py,validation.py,schemas.py}`.
Key invariants to preserve when touching this pipeline:

- `probe_id` (`IlmnID`) can repeat within a sample (up to ~43% duplicate rows
  observed) — it is annotation, never a row identity.
- Missing methylation is `NaN`/`<NA>`, never `0`; `methylated_count` and
  `unmethylated_count` are always jointly present or jointly missing.
- Probe-level aggregation **sums** `methylated_count`/`unmethylated_count`
  across duplicate rows (evidence-weighted) and never averages
  `methylation_fraction`.
- `scripts/data/build_canonical_methylation.py` never overwrites existing
  outputs without `--overwrite`, never touches raw files or hand-written
  audits under `$NEUROMETHYL_DATA_ROOT/registry/audit/`, and records
  per-sample failures (`status=failed`) in the manifest instead of aborting
  the whole run.

### Registry and provenance

`registry/{datasets.tsv,samples.tsv,provenance.tsv}` is the project-level
source of truth for dataset roles, patient-level sample identity
(`global_patient_id` must be stable across datasets for a recurring patient),
and generated-artifact provenance. Never commit protected patient
identifiers or private clinical metadata here. Splits must be
patient-disjoint (`docs/EXPERIMENT_POLICY.md`), not just run-disjoint —
repeated runs from the same tumor stay in one split, and the default cohort
roles (GSE90496=train, GSE109379=validation, GSE289246=primary ONT test,
RapidCNS2=secondary ONT test, GSE209865=legacy benchmark, nanoDx_EGA=raw-read
downsampling) must be re-validated after patient-level dedup, not assumed.

### External data layout

Real data lives outside the repo under `$NEUROMETHYL_DATA_ROOT`
(`docs/DATA_LAYOUT.md`): `raw/` is immutable, `processed/` is
dataset-specific standardized output, `derived/` is cross-dataset/
experiment-independent representations (e.g.
`derived/ont_native/coordinate_level/...` and
`derived/array_aligned/probe_level/...` for the two canonical Rapid-CNS2
representations), `benchmarks/` holds third-party competitor assets. Never
silently overwrite processed/derived data when a reference genome,
threshold, probe set, or caller changes.

## Conventions

- No datasets, patient data, checkpoints, or large generated artifacts get
  committed — those live under `$NEUROMETHYL_DATA_ROOT` or `outputs/`
  (gitignored).
- Avoid ambiguous names (`final`, `new`, `v2_final`, `test2`); use semantic
  experiment/config identifiers instead (e.g. `v0_binary_vs_beta_cov`).
- A PR touching data preprocessing or evaluation should state: affected
  dataset(s), whether sample inclusion changes, whether metrics stay
  comparable, and any migration/recompute requirements.
