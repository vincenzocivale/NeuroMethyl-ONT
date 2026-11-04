# NeuroMethyl-ONT

Nanopore-native classification of CNS tumors from DNA methylation.

NeuroMethyl-ONT asks whether keeping **continuous methylation values and
sequencing-depth evidence** (instead of collapsing to binary calls, as
Sturgeon and crossNN do) improves CNS tumor classification from Oxford
Nanopore data. Three linear classifiers share one architecture
(`nn.Linear(n_features, 91)`), one GSE90496-derived feature index and one
training schedule, and differ only in how features are built:

| Model | Features |
|---|---|
| **Binary** (formerly B0) | binary call, threshold 0.6 |
| **Beta** (formerly C0) | continuous beta minus the training-set mean |
| **Evidence** (formerly E1) | coverage-shrunk posterior beta minus the mean (prior strength `kappa`) |
| **Consensus** (formerly B0+E1) | average of Binary and Evidence softmax probabilities (recommended) |

Old names and how saved results stay readable: `docs/MODEL_NAMES.md`.

Sequencing depth only acts as confidence in the evidence (shrinkage), never
as an input feature, so a model cannot learn depth-to-batch-to-class
shortcuts. Models are benchmarked against Sturgeon, crossNN and MethyLYZR.
No clinical use is implied.

**Current question.** Can a classifier give *reliable* CNS tumour calls at low coverage, so that more
samples fit on one flow cell? Reliability is measured as accuracy, calibration and abstention against
sequencing depth, on the same samples and class set as the published competitors (Sturgeon, crossNN,
MethyLYZR, ...). Binary/Beta/Evidence below are the tools for that question.

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

## Install

```bash
pip install -e '.[dev]'
pytest && ruff check .
```

External data is never stored in this repository:

```bash
export NEUROMETHYL_DATA_ROOT=/path/to/CNSNanoporeData   # see docs/DATA_LAYOUT.md
```

## Pipeline

| Step | Entry point |
|---|---|
| Register datasets/samples | `scripts/data/validate_registry.py` |
| Download and prepare public data | `scripts/data/download_public_datasets.sh`, `prepare_and_audit_public_datasets.sh` |
| Build the GSE90496 reference and feature index | `scripts/data/prepare_geo_reference_inputs.py`, `build_array_reference.py` |
| Canonicalize ONT methylation (Rapid-CNS2, ONT-WGS, own samples) | `scripts/data/build_canonical_methylation.py`, `ingest_our_samples_pileup.py` |
| Train Binary / Beta / Evidence | `scripts/training/train_classifier.py` (configs in `configs/models/`) |
| Classify one sample (ensemble + per-probe explanation) | `scripts/inference/classify_sample.py` |
| CNS + leukemia two-stage mode (router, abstention) | same script with `--router-checkpoint` and `--alma-*` |
| Leukemia (ALMA) split, heads, router | `scripts/data/split_alma.py`, `scripts/training/train_alma.sh`, `train_router.py` |
| Run competitors | `scripts/competitors/` |
| Evaluate per cohort | `scripts/evaluation/` |
| Figures | `scripts/figures/` |

## Cohorts

| Dataset | Modality | Role |
|---|---|---|
| GSE90496 | Illumina 450K | training reference |
| GSE109379 | Illumina 450K | external validation |
| GSE289246 | ONT | primary ONT test |
| Rapid-CNS2 / ONT-WGS | ONT | secondary ONT test (real per-sample coverage) |
| GSE209865 | ONT-derived binary calls | legacy Sturgeon/nanoDx benchmark |
| Magibu pools, own MB/ETMR samples | ONT | additional low/high-coverage tests |
| ALMA discovery / nanopore | 450K-EPIC array / ONT | acute-leukemia head and domain router (`docs/ALMA_MILESTONE.md`) |

Splits are patient-disjoint; see `docs/EXPERIMENT_POLICY.md`.

## Repository layout

```text
configs/{data,models,experiments}/   versioned YAML defining every experiment
docs/                                methods, data layout, policies
registry/                            dataset / sample / provenance tables (no patient data)
scripts/{data,training,inference,competitors,evaluation,figures}/
src/neuromethyl_ont/{data,models,training,evaluation}/
tests/
third_party/                         vendored competitor code (see third_party/README.md)
```

## Documentation

- [`docs/EVIDENCE_LINEAR_V1.md`](docs/EVIDENCE_LINEAR_V1.md) — Binary/Beta/Evidence specification (features, training, evaluation).
- [`docs/ALMA_MILESTONE.md`](docs/ALMA_MILESTONE.md) — acute-leukemia head, domain router, results and limits.
- [`docs/PRODUCTION_MODEL.md`](docs/PRODUCTION_MODEL.md) — why the Consensus ensemble, with cohort results and caveats.
- [`docs/MODEL_NAMES.md`](docs/MODEL_NAMES.md) — old names (B0/C0/E1/B0+E1) to new, and how saved results stay readable.
- [`docs/RAPIDCNS2_CANONICAL_PREPROCESSING.md`](docs/RAPIDCNS2_CANONICAL_PREPROCESSING.md) — canonical ONT representations.
- [`docs/GSE109379_IDAT_RECONSTRUCTION.md`](docs/GSE109379_IDAT_RECONSTRUCTION.md) — GSE109379 rebuilt from raw IDATs.
- [`docs/DATA_LAYOUT.md`](docs/DATA_LAYOUT.md), [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), [`docs/EXPERIMENT_POLICY.md`](docs/EXPERIMENT_POLICY.md), [`CONTRIBUTING.md`](CONTRIBUTING.md).

## License

No license has been selected yet; replace `LICENSE_PLACEHOLDER.md` with a `LICENSE` before public release.
Third-party code and model assets remain under their own licenses.
