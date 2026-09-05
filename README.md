# NeuroMethyl-ONT

**Nanopore-native methylation classification of CNS tumors.**

NeuroMethyl-ONT is a research codebase for developing and benchmarking CNS tumor classifiers that exploit methylation information from Oxford Nanopore sequencing, with emphasis on **full-run diagnostic performance**, **continuous methylation**, **coverage-aware modelling**, and eventual **genome-wide** inference.

## Project principles

- **Reproducibility first**: every transformation from source data to benchmark output should be scripted and configurable.
- **No hidden data state**: raw, processed, derived, and experimental outputs are kept conceptually separate.
- **Patient-level provenance**: dataset origin, patient/sample identifiers, duplicate relationships, platform, and taxonomy are tracked centrally.
- **Frozen evaluation cohorts**: training, validation, primary ONT test, secondary ONT test, and legacy benchmarks must remain distinct.
- **Nanopore-native design**: avoid unnecessarily collapsing sequencing data into array-like binary representations when richer information is available.

## Repository layout

```text
NeuroMethyl-ONT/
├── configs/                 # Versioned data/model/experiment configs
├── docs/                    # Architecture, data and experiment documentation
├── registry/                # Dataset/sample registry templates (no patient data committed)
├── scripts/                 # Thin executable entry points
├── src/neuromethyl_ont/     # Importable Python package
├── tests/                   # Unit and smoke tests
├── outputs/                 # Local generated outputs (mostly gitignored)
├── pyproject.toml
└── README.md
```

Large datasets are **not stored in this repository**. Set the data root with:

```bash
export NEUROMETHYL_DATA_ROOT=/path/to/CNSNanoporeData
```

See [`docs/DATA_LAYOUT.md`](docs/DATA_LAYOUT.md) for the expected external data layout.

## Intended cohort roles

| Dataset | Modality | Intended role |
|---|---|---|
| GSE90496 | Illumina 450K | training reference |
| GSE109379 | Illumina 450K | external validation / model selection |
| GSE289246 | ONT / sequencing methylation | primary ONT test |
| RapidCNS2 | ONT / sequencing methylation | secondary ONT test |
| GSE209865 | ONT-derived binary calls | legacy Sturgeon/nanoDx benchmark |
| nanoDx_EGA | raw ONT | read-level downsampling / scaling experiments |

These roles are defaults, not a substitute for a finalized leakage audit. Before any training run, populate and validate the registries in `registry/`.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
```

Validate the local project/data configuration:

```bash
python scripts/data/validate_registry.py
```

## Development workflow

1. Register source datasets and samples.
2. Freeze patient-level train/validation/test provenance.
3. Download data outside the repository.
4. Create deterministic preprocessing configs.
5. Write derived artifacts under the external data root.
6. Run experiments using versioned configs.
7. Save metrics/config snapshots under `outputs/experiments/`.

## Documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — codebase boundaries and design rules.
- [`docs/DATA_LAYOUT.md`](docs/DATA_LAYOUT.md) — external dataset organization.
- [`docs/RAPIDCNS2_CANONICAL_PREPROCESSING.md`](docs/RAPIDCNS2_CANONICAL_PREPROCESSING.md) — Rapid-CNS²/ONT-WGS parser, canonical coordinate-level/probe-level representations, and the `build_canonical_methylation.py` CLI.
- [`docs/EXPERIMENT_POLICY.md`](docs/EXPERIMENT_POLICY.md) — split hygiene, benchmark policy and reporting rules.
- [`docs/ROADMAP.md`](docs/ROADMAP.md) — staged research plan.
- [`docs/EVIDENCE_LINEAR_V1.md`](docs/EVIDENCE_LINEAR_V1.md) — the V0 representation ablation (B0/C0/E1): exact feature/training/evaluation spec.
- [`docs/GSE109379_IDAT_RECONSTRUCTION.md`](docs/GSE109379_IDAT_RECONSTRUCTION.md) — why and how GSE109379 is reconstructed from raw IDATs (GEO's precomputed matrix is truncated) via the official MNPpreprocessIllumina pipeline.
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — development conventions.

## Status

Initial research scaffold. No clinical use is implied or supported.
