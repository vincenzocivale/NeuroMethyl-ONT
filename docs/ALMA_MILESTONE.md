# ALMA leukemia classifier milestone

Design spec for a **separate** classifier milestone built on the ALMA acute
leukemia methylation atlas (Marchi et al., *Nature Communications* 16, 6961
(2025), doi:10.1038/s41467-025-62005-4). This document is the source of
truth for `src/neuromethyl_ont/data/alma.py`, `registry/datasets.tsv` rows
`ALMA_discovery`/`ALMA_nanopore`, and any `configs/models/alma_*.yaml` /
`configs/experiments/alma_*.yaml` files.

## 1. Relationship to the CNS milestone

ALMA classifies acute leukemias (AML, ALL, MDS, APL, MPAL per WHO 2022), not
CNS tumors. It does **not** extend the GSE90496-derived 91-class CNS
taxonomy used by B0/C0/E1 (`docs/EVIDENCE_LINEAR_V1.md`), and does not share
a reference cohort, probe index, or class list with it. Nothing in this
milestone should be merged into the CNS models' feature index or training
data. Treat this as a second, independent classifier family sharing only
the repository's canonical-representation conventions
(`docs/ARCHITECTURE.md`).

## 2. Data

Two public files from the GitHub `f-marchi/ALMA` v0.2.0 release, plus one
Nature Communications supplementary file, all under
`$NEUROMETHYL_DATA_ROOT/datasets/ALMA/`:

- `raw/3314samples_331556cpgs_bvalues.pkl.gz` -- discovery/training cohort.
  3314 samples x 331556 CpGs, 450K/EPIC/EPICv2 array betas, batch-corrected.
  Index = anonymized/pseudonymous `sample_id` (real Sentrix IDs for
  GEO-sourced batches, `<uuid>_noid` for GDC/TARGET/TCGA-sourced batches);
  one non-CpG column, `Batch`, records the source cohort
  (GSE49031/GSE190931/GSE124413/GSE159907/GSE152710/GSE147667/GSE133986,
  GDC_TARGET-AML, GDC_TARGET-ALL, GDC_TCGA-AML).
- `raw/20samples_nanopore_866857cpgs.csv.gz` -- ONT evaluation cohort. CpGs
  as rows, 20 samples (`alma_<n>_BM` / `alma_<n>_PB`) as columns -- the
  transpose of the discovery layout; `load_nanopore_betas` transposes it
  back to sample-indexed rows.
- `raw/clinical_data_MOESM4.xlsx` -- Nature Communications supplementary
  file, sheet `indexed_clinical_data`: clinical/subtype labels for 3527
  samples (the 3314 discovery samples plus other cohorts not covered by the
  public beta release).

### 2.1 Discovery label join (resolved)

The clinical Excel's `sample_id` index and the discovery pickle's index are
the *same* identifiers -- joining on `sample_id` (never on row position: the
published Excel's row order is shuffled relative to the pickle, evidently
for de-identification) gives a clean match. This was verified by checking
that the joined `Batch` column from each source agrees for every one of the
3314 discovery samples (0 mismatches).

`load_discovery_dataset()` performs this join and raises if the check ever
fails again (e.g. after a data refresh). The precomputed label table lives
at `datasets/ALMA/processed/discovery_clinical_labels.parquet`
(`sample_id`, `Batch`, `WHO 2022 Diagnosis`, `AL Epigenomic Subtype`,
`AL Epigenomic Subtype_int`, `Train-Test`, `Sex`, `Patient_ID`,
`Age (years)`, `Tissue Type`, `Sample Type`). 2987/3314 samples have a
non-null `AL Epigenomic Subtype`; the remaining 327 are unlabeled in the
source publication (mostly ambiguous/excluded cases) and must be dropped or
handled explicitly, never silently treated as a 29th class.

### 2.2 Nanopore label gap (open)

No public file maps the 20 `alma_<n>_BM/PB` nanopore sample IDs to a
subtype label: the string does not appear in any sheet of the clinical
Excel (checked exhaustively, including the per-figure data tabs), and the
authors' own `source/3_nanopore_test.ipynb` notebook only runs
basecalling -> bedMethyl -> `alma-classifier` prediction, without
referencing ground-truth labels for these samples. Until this is resolved
(most likely by transcribing the article's own patient description table
by hand), `load_nanopore_betas()` returns betas only and this cohort can be
used for qualitative/prediction-only checks, not as a labeled evaluation
set.

## 3. Class taxonomy

28 `AL Epigenomic Subtype` classes appear in the discovery labels (see
`registry/datasets.tsv` note and `discovery_clinical_labels.parquet` for
the per-class counts, which range from 933 samples for the most common
GSE49031 batch down to single digits for some rare subtypes -- expect a
long-tailed class distribution, unlike GSE90496's more even 91-class
split). This is now frozen as a feature/class index, built by
`scripts/data/build_alma_reference.py` (reusing
`neuromethyl_ont.data.array_reference`'s generic build/save/align
machinery, the same code B0/C0/E1 use for GSE90496 -- but as an entirely
separate index/checksum) from the 2987 labeled discovery samples, dropping
the 327 unlabeled ones. Output at
`$NEUROMETHYL_DATA_ROOT/derived/reference/ALMA_discovery/`: all
331556/331556 probes passed the `min_valid_fraction=0.95` / zero-variance /
duplicate-id filters unchanged (the discovery matrix ships pre-filtered and
batch-corrected).

## 4. Architecture: specialist heads + domain router

CNS (91 classes) and leukemia (28 classes) are kept as separate specialist
heads with separate references and `mu`; a small router decides which head to
run and abstains when unsure. The CNS head is frozen, so every CNS result is
unaffected by adding leukemia.

- **Split** (`scripts/data/split_alma.py`, `alma_split_v1.tsv`): patient-disjoint
  (by `Patient_ID`) and stratified by subtype, 2134 / 430 / 423 samples
  (train/val/test). The public release has no held-out test (all samples are
  `Train Sample`), and 2987 samples come from 2776 patients.
- **Reference** (`build_alma_reference.py --split-file`): probe filter and `mu`
  from the train split only; val/test exported under
  `derived/evaluation/ALMA/`.
- **Heads** (`scripts/training/train_alma.sh`): B0/C0/E1 with the CNS recipe.
- **Router** (`scripts/training/train_router.py`): linear, B0-style binary
  features on the 331,433 probes shared with GSE90496, random probe masking.

## 5. Results (single seed, single split)

ALMA test, n=423 (`scripts/evaluation/evaluate_alma.py`):

| Model | Acc. | Bal. acc. | Macro-F1 | NLL | ECE |
|---|--:|--:|--:|--:|--:|
| B0 | 83.7% | 72.1% | 71.6% | 2.57 | 0.147 |
| C0 | 77.5% | 65.9% | 65.3% | 1.05 | 0.059 |
| E1 (full-array limit) | 77.5% | 66.9% | 66.4% | 1.22 | 0.109 |
| B0+E1 | 84.2% | 72.4% | 72.1% | 0.95 | 0.062 |

The ensemble matches B0 in accuracy (+2 samples, not significant) and is much
better calibrated. C0/E1 are worse than B0 here; hyperparameters were not
tuned for the continuous representations, so this is not evidence against them.

Router (`scripts/evaluation/evaluate_router.py`, threshold 0.9): 100% domain
accuracy on ALMA test, the 20 ALMA nanopore samples, ONT-WGS and GSE109379;
Rapid-CNS2 299/300 correct (one immune-like sample called CTRL by every CNS
model is routed to leukemia). The 8 own low-coverage samples are all correct by
argmax but abstain at 0.9 (p_CNS 0.58-0.79). Widening the training-time masking
to 97% (`router_sparse_seed17`) did not change this.

## 6. Two-stage inference

`scripts/inference/classify_sample.py` with `--router-checkpoint` and the
`--alma-*` arguments runs the router and then the selected head's B0+E1
ensemble (logic in `src/neuromethyl_ont/inference/two_stage.py`). Below
`--router-threshold` (default 0.9) the sample is `undetermined`: no head is
selected and both heads' top-3 are listed for transparency, so a low-confidence
domain is never turned into a diagnosis. Smoke-tested on real Rapid-CNS2 and
ONT-WGS samples (routed to CNS), one own low-coverage sample (undetermined) and
one ALMA test sample with *simulated* 3x counts (routed to leukemia, correct
subtype). No real ONT leukemia counts were available for an end-to-end run.

## 7. Limitations

- No subtype-labelled nanopore data for leukemia: the 20 nanopore samples only
  validate the *domain* routing. Subtype accuracy is array-only.
- Several rare subtypes have under 3 test samples; per-class metrics are noisy.
- One seed, one split. E1 is evaluated in its full-array limit (no counts).
- The single 119-class softmax ablation has not been run.
