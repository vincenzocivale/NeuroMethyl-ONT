# Experiment policy

## Default cohort roles

- **GSE90496** — training reference
- **GSE109379** — validation/model selection
- **GSE289246** — primary ONT test
- **RapidCNS2** — secondary ONT test
- **GSE209865** — legacy Sturgeon/nanoDx benchmark
- **nanoDx_EGA** — raw-read/downsampling analyses

These assignments must be revalidated after patient-level deduplication.

## Leakage prevention

- Splits are patient-disjoint, not merely run-disjoint.
- Repeated runs from one tumor stay in one split.
- Cohorts reused across publications must be represented once in the global provenance table.
- External test cohorts are never used for hyperparameter/model selection.

## Minimum reporting

For each final benchmark report at least:

- accuracy,
- balanced accuracy,
- macro-F1,
- per-family/per-class performance,
- calibration/confidence,
- abstention/coverage curve where applicable,
- number of evaluable samples,
- missingness / observed CpGs,
- sequencing-depth or downsampling condition.

## Core V0 ablation

Hold the CpG feature space constant and compare:

1. binary methylation,
2. continuous beta,
3. beta + coverage/evidence.

The purpose is to isolate whether information discarded by current classifiers is actually useful before introducing a more complex genome-wide architecture.
