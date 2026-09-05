# GSE109379: reconstructing the validation matrix from raw IDATs

## Why

`GSE109379_processed_data.txt.gz` (GEO's precomputed beta-value matrix for
the validation cohort in docs/EVIDENCE_LINEAR_V1.md) is truncated:

- Expected ~428,799 CpG rows (matching GSE90496, same GPL13534 platform);
  the file has only 42,383, and the last row cuts off mid-float with 356 of
  the expected 2,209 columns.
- The gzip container itself is valid (`gzip -t` passes) and its MD5 matches
  `registry/source_md5.tsv` and GEO's currently-hosted file size -- so this
  is an upstream malformed/truncated file, not a local download failure, and
  re-downloading it would not fix it.
- `GSE90496_beta.txt.gz` was checked the same way and is intact: 428,799
  rows, no ragged rows, all 5,603 columns.

Rather than validate on ~10% of the probe space, GSE109379 is reconstructed
from its raw IDATs (`GSE109379_RAW.tar`, ~9.7 GB,
`ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE109nnn/GSE109379/suppl/GSE109379_RAW.tar`)
using the same preprocessing the original array was actually run through.

## Pipeline

```text
GSE109379_family.soft.gz
    -> export_idat_sample_manifest.py   (GSM, methylation class, IDAT basename)
    -> idat_to_beta_mnp.R               (minfi::read.metharray, MNPpreprocessIllumina,
                                          detectionP; see below)
    -> validate_idat_reconstruction.py  (compare against the ~42k intact GEO rows)
```

`idat_to_beta_mnp.R` runs, per sample:

1. `minfi::read.metharray` -> raw `RGChannelSet`.
2. `minfi::detectionP(RGset)` on the **raw**, pre-normalization set (matches
   how GEO's own "Detection Pval" column is computed independent of
   downstream normalization).
3. `MNPpreprocessIllumina(RGset, bg.correct = TRUE, normalize = "controls",
   ref = 10000)` -- the official DKFZ preprocessing from
   [mwsill/mnp_training](https://github.com/mwsill/mnp_training) (vendored in
   `third_party/mnp_training/`, MIT license), the code Capper et al. actually
   used to build GSE90496/GSE109379. It is `minfi::preprocessIllumina` with
   one change: every array's control-probe intensities are scaled to a
   **fixed** reference of 10,000, rather than minfi's default of scaling to
   the first/reference sample's own control intensity -- so, unlike vanilla
   `preprocessIllumina`, the result for a given sample does not depend on
   which other samples are processed alongside it.
4. `beta = Meth / (Meth + Unmeth + 100)` (minfi/Illumina convention).

Output: `{meth,unmeth,beta,detection_p}.f8` (flat float64 binaries) +
`probe_ids.txt` / `sample_ids.txt`, read back via
`neuromethyl_ont.data.idat_reconstruction.load_mnp_preprocessed`.

## Environment note

This box's `matrixStats` (>= 1.2.0) made the legacy `useNames = NA` default
an error, but the installed `minfi`/`MatrixGenerics` (pinned to R 4.2.1 /
Bioconductor 3.16) still pass it from `detectionP`'s internal
`colMedians`/`colMads` calls. `idat_to_beta_mnp.R` sets
`options(matrixStats.useNames.NA = "deprecated")` to restore the old (still
correct) behavior with a warning instead of aborting -- see
`?matrixStats::matrixStats.options`.

## Validation result (docs/EVIDENCE_LINEAR_V1.md's "do not touch external
test sets" applies here too -- this only touches GSE109379, the validation
cohort)

10-sample pilot, reconstructed beta vs. the matching intact GEO rows
(`scripts/data/validate_idat_reconstruction.py`, defaults: min Pearson r
0.98, max median |diff| 0.02):

```json
{
  "n_common_probes": 42382,
  "n_compared_values": 423820,
  "pearson_r": 0.9989,
  "median_abs_diff": 0.0082,
  "max_abs_diff": 0.586,
  "fraction_within_0_05": 0.968
}
```

PASS. The tail of larger per-probe deviations (~3% of values beyond 0.05) is
consistent with minor cross-version differences in background correction
between this run's minfi and whatever produced GEO's original matrix in
2018, not a pipeline defect -- see the full-cohort run's report
(`datasets/GSE109379/processed/mnp_reconstruction/validation_report.json`)
for the number this project actually used.

## Sample manifest

`export_idat_sample_manifest.py` derives, per GSM, the exact IDAT basename
from the SOFT file's `!Sample_supplementary_file` lines (never guessed from
extracted filenames), and fails loudly if a sample's Grn/Red files disagree
on basename. `n_samples_limit` in `idat_to_beta_mnp.R` lets a subset be
reconstructed first (as above) before committing to the full cohort.

## Scope

This reconstruction touches **only** GSE109379 (training-time model
selection / validation cohort). GSE209865, Rapid-CNS2, ONT-WGS and GSE289246
are untouched. GSE90496 (training) is confirmed intact and is used as-is for
now; reconstructing it from IDAT with this same pipeline is planned as a
follow-up for the definitive (non-sanity-check) experiment, once GSE109379's
reconstruction is the established, validated approach.
