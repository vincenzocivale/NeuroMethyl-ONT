# Registry

These files define the project-level source of truth for datasets, samples and generated artifacts.

## `datasets.tsv`
One row per external dataset/source. Describes intended role and source format.

## `samples.tsv`
One row per sample/run after metadata harmonization. `global_patient_id` must be stable across datasets when the same patient is known to recur.

## `provenance.tsv`
Tracks generated processed/derived artifacts back to source data, code version and configuration.

Do not commit protected patient identifiers or private clinical metadata to Git.

## Status vocabulary (datasets.tsv)
Not enforced by the validator; used so far: `planned` (not on disk), `downloaded` (raw/processed present in the data root), `labeled` / `unlabeled` (ready, with / without ground-truth labels).
