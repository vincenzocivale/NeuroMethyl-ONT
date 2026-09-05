# Contributing

## Non-negotiable conventions

1. **Do not commit biomedical datasets, patient-level private data, checkpoints, or large generated artifacts.**
2. Every experiment must be reproducible from a versioned config plus a documented dataset registry state.
3. Patient-level splits take precedence over sample/run-level splits.
4. `scripts/` contains thin entry points; reusable logic belongs in `src/neuromethyl_ont/`.
5. Raw source files are immutable. Reprocessing creates new derived artifacts rather than overwriting raw data.
6. New derived datasets require a provenance record describing source files, code/config version, reference genome, and transformation.
7. Avoid ambiguous names such as `final`, `new`, `v2_final`, or `test2`. Prefer semantic names and explicit version/config identifiers.

## Before committing

```bash
ruff check .
pytest
```

## Naming

- Python modules/functions: `snake_case`
- Classes: `PascalCase`
- Experiment IDs: semantic, e.g. `v0_binary_vs_beta_cov`
- Dataset IDs: canonical accessions or stable project names

## Pull requests

A PR changing data preprocessing or evaluation must state:

- affected dataset(s),
- whether sample inclusion changes,
- whether metrics are no longer directly comparable,
- migration/recompute requirements.
