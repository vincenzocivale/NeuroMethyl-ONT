# Experiment policy

## Default cohort roles

- **GSE90496** — training reference
- **GSE109379** — validation/model selection
- **GSE289246** — primary ONT test
- **RapidCNS2** — secondary ONT test
- **GSE209865** — legacy Sturgeon/nanoDx benchmark
- **nanoDx_EGA** — raw-read/downsampling analyses

These assignments must be revalidated after patient-level deduplication.

## Label/class audit tooling

`scripts/competitors/extract_sturgeon_classes.py` and
`inspect_sturgeon_model_zip.py` (migrated from the sibling
`cns-io-genomics-data-audit` project) answer "which Sturgeon classes can a
given cohort actually support": `extract_sturgeon_classes.py` parses a
Sturgeon model zip's raw + merged operational class list,
`inspect_sturgeon_model_zip.py` previews an unfamiliar model zip before
that, and `src/neuromethyl_ont/data/label_audit/{gdc,cbioportal}.py` are
thin API clients for cross-checking TCGA/IPD-Brain/DBTA case metadata
against those classes. Run `extract_sturgeon_classes.py` once per Sturgeon
model release, not per experiment.

`src/neuromethyl_ont/evaluation/paired_bootstrap.py` adds
significance testing on top of the existing point-estimate comparison
scripts (`scripts/evaluation/evaluate_rapidcns2.py`,
`evaluate_magibu_pools.py`): given two models' per-sample class
probabilities on the same cohort, `compare_paired` reports a stratified
paired-bootstrap 95% CI on the accuracy/NLL/Brier/macro-F1 delta, not just
the point estimate.

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
