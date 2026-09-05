# Architecture

## Boundary rules

The codebase is deliberately layered:

```text
source data
   ↓
data readers / harmonization
   ↓
canonical sample representations
   ↓
models
   ↓
training / inference
   ↓
evaluation
   ↓
experiment artifacts
```

### `src/neuromethyl_ont/data`
Owns schemas, dataset adapters, preprocessing and representation conversion. It must not contain model-specific training loops.

### `src/neuromethyl_ont/models`
Owns model definitions only. Models consume canonical tensors/records and should not know GEO/EGA filenames.

### `src/neuromethyl_ont/training`
Owns optimization, checkpointing and experiment execution.

### `src/neuromethyl_ont/evaluation`
Owns metrics, confidence/abstention analysis, coverage-scaling evaluation and competitor adapters.

### `src/neuromethyl_ont/utils`
Small infrastructure helpers. Domain logic should not accumulate here.

### `scripts/`
Thin CLI-oriented wrappers. If a script grows substantial logic, move that logic into `src/` and test it.

## Canonical representations

The project should support progressively richer methylation representations:

1. **Binary array-like**: `(probe_id, state)`
2. **Continuous**: `(locus, beta)`
3. **Coverage-aware**: `(locus, methylated_count, unmethylated_count, beta, coverage)`
4. **Nanopore-native**: variable-size locus/read sets with genomic context and quality metadata

Keeping conversion explicit is essential for fair ablations.
