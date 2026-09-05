# Vendored: mwsill/mnp_training

`MNPprocessIDAT_functions.R` is an unmodified copy of
[`R/MNPprocessIDAT_functions.R`](https://github.com/mwsill/mnp_training/blob/ece7262015b896d31078d8125091c83c06158d59/R/MNPprocessIDAT_functions.R)
from [mwsill/mnp_training](https://github.com/mwsill/mnp_training)
(commit `ece7262015b896d31078d8125091c83c06158d59`), the official preprocessing
code accompanying Capper et al., *DNA methylation-based classification of
central nervous system tumours*, Nature 2018 (the GSE90496/GSE109379
reference/validation cohorts this project trains and validates on).

It defines `MNPpreprocessIllumina()` / `MNPnormalize.illumina.control()`: a
modified `minfi::preprocessIllumina` that normalizes every array's control
probes to a **fixed** reference intensity (default 10,000) instead of minfi's
default of normalizing to the first/reference sample's own control intensity.
`scripts/data/idat_to_beta_mnp.R` sources this file verbatim and calls it with
that same default (`ref = 10000`) -- see docs/EVIDENCE_LINEAR_V1.md and
`docs/GSE109379_IDAT_RECONSTRUCTION.md` for why this project reconstructs
GSE109379 (and, later, GSE90496) from raw IDATs with this exact pipeline
rather than GEO's precomputed supplementary matrix.

Licensed MIT (`LICENSE`, copyright (c) 2020 Martin Sill), reproduced
unmodified per that license's attribution requirement.
