# Rapid-CNS² / ONT-WGS canonical methylation preprocessing

This describes the pipeline that turns raw Rapid-CNS² / MNP-Flex-derived
methylation files into the two canonical representations used throughout the
project. It implements the preprocessing brief in full; this doc is the
short version for contributors.

## Source data

Two cohorts under `RapidCNS2/raw/`:

- **ONT_WGS** (39 files, `ONT_WGS_<n>.txt`, gap at `ONT_WGS_15`) — median mean
  coverage ≈ 29.6x.
- **Rapid_CNS2** (300 files, `ID_<n>.bed.txt`, gap at `ID_117`) — median mean
  coverage ≈ 2.0x.

Both are the output of an MNP-Flex-style `bedtools intersect` between a
methylation BED and the array probe BED, which is why the same `probe_id`
(`IlmnID`) can appear at multiple genomic coordinates within one sample
(observed up to ~43% duplicate probe rows in the worst Rapid-CNS2 sample).
`probe_id` is therefore treated as annotation, never as a row identity.

## Modules

- `src/neuromethyl_ont/data/schemas.py` — column names/dtypes for the two
  canonical record types (`CoordinateMethylationRecord`, `ProbeMethylationRecord`).
- `src/neuromethyl_ont/data/rapidcns2.py` — `read_rapidcns2_sample(path)`:
  header-aware, schema-aware parser. Handles headerless files, an optional
  `mod` column, tab- or space-separated fields, quoted headers/values, and
  explicit missing tokens (`NA`/`NaN`/`.`/`null`, case-insensitive). Raises
  `SchemaDetectionError` rather than guessing when no known layout is found.
- `src/neuromethyl_ont/data/canonical.py` — `to_coordinate_level(df, sample_id)`
  and `aggregate_to_probe_level(coord_df)`. Probe aggregation sums
  `methylated_count`/`unmethylated_count` across all rows sharing a
  `probe_id` (evidence-weighted); it never averages `methylation_fraction`.
- `src/neuromethyl_ont/data/validation.py` — `validate_coordinates`,
  `validate_counts`, `validate_beta_range`, `validate_probe_aggregation`.
  Each returns a list of violation strings (empty = valid); `assert_valid`
  raises `ValidationError` if the list is non-empty.

## Missingness

A missing methylation percentage produces `methylation_fraction = NaN` and
`methylated_count = unmethylated_count = <NA>` — never `0`. Coverage can be
present while methylation is missing (coverage measured, percentage not
reported); that is a valid, non-inconsistent state. `methylated_count`/
`unmethylated_count` are always jointly present or jointly missing.

## Reconstructing integer counts

Published percentages are rounded to two decimals, so:

```python
beta = methylation_percentage / 100
m = round(coverage * beta)   # clipped to [0, coverage]
u = coverage - m
```

At low coverage this rounding can shift `m / coverage` away from `beta` by up
to `0.5 / coverage`; `validate_beta_range` scales its tolerance with coverage
accordingly rather than using one fixed epsilon.

## CLI

```bash
python scripts/data/build_canonical_methylation.py "$DATA_ROOT" \
    --dataset RapidCNS2 \
    [--cohort ONT_WGS|Rapid_CNS2] [--sample ID_1] [--overwrite] [--dry-run]
```

- Writes coordinate-level Parquet to
  `$DATA_ROOT/derived/ont_native/coordinate_level/RapidCNS2/<cohort>/<sample_id>.parquet`.
- Writes probe-level Parquet to
  `$DATA_ROOT/derived/array_aligned/probe_level/RapidCNS2/<cohort>/<sample_id>.parquet`.
- Updates (merges into, does not overwrite unrelated rows of)
  `$DATA_ROOT/registry/derived/canonical_methylation_manifest.tsv`.
- Does not overwrite existing outputs unless `--overwrite` is passed.
- A sample that fails schema detection or an invariant check is logged as an
  error and recorded with `status=failed` in the manifest; it does not abort
  the run for other samples.
- Logs to `outputs/logs/data/build_canonical_methylation_<timestamp>.log`.
- Never modifies raw files or the existing hand-written audits under
  `$DATA_ROOT/registry/audit/`.

## What this pipeline deliberately does not do

No label harmonization, no `tumor_type` → `methylation_class` mapping, no
training, no read-level downsampling, no CNV integration. See the project
brief for the full list and rationale — those are separate, later tasks.
