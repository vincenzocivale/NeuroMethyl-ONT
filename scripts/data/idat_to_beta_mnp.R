#!/usr/bin/env Rscript
# IDAT -> {Meth, Unmeth, Beta, DetectionP} using the official DKFZ/mnp_training
# preprocessing (MNPpreprocessIllumina, fixed control-reference intensity
# 10,000) -- see docs/EVIDENCE_LINEAR_V1.md and
# docs/GSE109379_IDAT_RECONSTRUCTION.md for why this exists: GEO's
# precomputed GSE109379_processed_data.txt.gz is truncated mid-download.
#
# MNPpreprocessIllumina (third_party/mnp_training, vendored from
# mwsill/mnp_training, MIT) is a minfi::preprocessIllumina variant that
# normalizes every array's control probes to a FIXED reference intensity
# instead of minfi's default of normalizing to the first/reference sample's
# own control intensity -- this is what the actual GSE90496/GSE109379 arrays
# were processed with (Capper et al., Nature 2018).
#
# Detection P-values are computed on the RAW (pre-normalization) RGChannelSet
# via minfi::detectionP, matching how GEO's own "Detection Pval" column is
# computed independent of any downstream normalization choice.
#
# beta = Meth / (Meth + Unmeth + 100)  -- minfi/Illumina-style, offset = 100.
#
# Output (four flat float64 binaries + shared row/column id files, so a
# 1104-sample x ~485k-probe array doesn't need a text round-trip):
#   <output_dir>/probe_ids.txt      one probe id per line (row order)
#   <output_dir>/sample_ids.txt     one GSM per line (column order)
#   <output_dir>/meth.f8            Meth,  column-major flatten (probes x samples)
#   <output_dir>/unmeth.f8          Unmeth, same layout
#   <output_dir>/beta.f8            Beta,   same layout
#   <output_dir>/detection_p.f8     DetectionP, same layout
#   <output_dir>/preprocess_metadata.txt   preprocessMethod + package versions
#
# A column-major flatten of an (n_probes x n_samples) R matrix is byte-for-byte
# a row-major flatten of its (n_samples x n_probes) transpose, so the Python
# side reads with plain `np.fromfile(path, dtype="<f8").reshape(n_samples, n_probes)`
# -- no transpose needed on either side.
#
# Usage:
#   Rscript scripts/data/idat_to_beta_mnp.R \
#       <idat_dir> <manifest_tsv> <output_dir> [n_samples_limit]
#
# <manifest_tsv> is produced by scripts/data/export_idat_sample_manifest.py
# (columns: gsm, sample_column, methylation_class, sample_title, idat_basename).
# [n_samples_limit], if given, processes only the first N manifest rows -- for
# a fast correctness check before committing to the full cohort.

# Environment skew workaround: this box's matrixStats (>= 1.2.0) made the old
# useNames = NA default an error, but the installed MatrixGenerics/minfi still
# pass it (e.g. from minfi::detectionP's internal colMedians/colMads calls).
# "deprecated" restores the old (correct) NA behavior with a warning instead
# of aborting -- see ?matrixStats::matrixStats.options.
options(matrixStats.useNames.NA = "deprecated")

suppressPackageStartupMessages({
  library(minfi)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  stop("Usage: idat_to_beta_mnp.R <idat_dir> <manifest_tsv> <output_dir> [n_samples_limit]")
}
idat_dir <- args[1]
manifest_tsv <- args[2]
output_dir <- args[3]
n_limit <- if (length(args) >= 4) as.integer(args[4]) else NA_integer_

script_dir <- dirname(sub("--file=", "", grep("--file=", commandArgs(trailingOnly = FALSE), value = TRUE)))
repo_root <- normalizePath(file.path(script_dir, "..", ".."))
source(file.path(repo_root, "third_party", "mnp_training", "MNPprocessIDAT_functions.R"))

dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

log_msg <- function(...) cat(sprintf("[%s] %s\n", format(Sys.time(), "%H:%M:%S"), sprintf(...)))

# ---- 1. Manifest + locate IDAT files -------------------------------------
manifest <- read.delim(manifest_tsv, stringsAsFactors = FALSE)
required_cols <- c("gsm", "idat_basename")
missing_cols <- setdiff(required_cols, colnames(manifest))
if (length(missing_cols) > 0) {
  stop("manifest missing column(s): ", paste(missing_cols, collapse = ", "))
}
if (!is.na(n_limit)) {
  manifest <- manifest[seq_len(min(n_limit, nrow(manifest))), ]
}
log_msg("Manifest: %d samples", nrow(manifest))

idat_files <- list.files(idat_dir, recursive = TRUE, pattern = "_(Grn|Red)\\.idat(\\.gz)?$", full.names = TRUE)
if (length(idat_files) == 0) stop("No _Grn/_Red .idat(.gz) files found under: ", idat_dir)
# Map basename (path with channel/extension stripped) -> full path prefix minfi expects.
idat_prefix_of <- sub("_(Grn|Red)\\.idat(\\.gz)?$", "", idat_files)
names(idat_prefix_of) <- basename(idat_prefix_of)
idat_prefix_of <- idat_prefix_of[!duplicated(names(idat_prefix_of))]

missing_basenames <- setdiff(manifest$idat_basename, names(idat_prefix_of))
if (length(missing_basenames) > 0) {
  stop(sprintf(
    "%d manifest sample(s) have no matching IDAT files under %s, e.g.: %s",
    length(missing_basenames), idat_dir, paste(head(missing_basenames, 5), collapse = ", ")
  ))
}
basenames <- unname(idat_prefix_of[manifest$idat_basename])

# ---- 2. Read raw intensities ----------------------------------------------
log_msg("Reading %d IDAT pairs ...", length(basenames))
RGset <- read.metharray(basenames, verbose = TRUE, force = TRUE)
sampleNames(RGset) <- manifest$gsm
log_msg("RGChannelSet: %d probes x %d samples, array=%s", nrow(RGset), ncol(RGset), annotation(RGset)[["array"]])

# ---- 3. Detection P-values, on the RAW (pre-normalization) set ------------
log_msg("Computing detectionP ...")
detection_p <- detectionP(RGset)

# ---- 4. MNPpreprocessIllumina: fixed-reference control normalization ------
log_msg("Running MNPpreprocessIllumina (ref = 10000) ...")
Mset <- MNPpreprocessIllumina(RGset, bg.correct = TRUE, normalize = "controls", ref = 10000)
rm(RGset)
gc()

meth <- getMeth(Mset)
unmeth <- getUnmeth(Mset)
beta <- meth / (meth + unmeth + 100)
probe_ids <- rownames(meth)
sample_ids <- colnames(meth)

stopifnot(identical(rownames(detection_p), probe_ids) || nrow(detection_p) == nrow(meth))
stopifnot(identical(colnames(detection_p), sample_ids))
# detectionP's probe order can differ from preprocessRaw's; align defensively.
detection_p <- detection_p[probe_ids, sample_ids, drop = FALSE]

log_msg("Output matrices: %d probes x %d samples", length(probe_ids), length(sample_ids))

# ---- 5. Write outputs -------------------------------------------------
writeLines(probe_ids, file.path(output_dir, "probe_ids.txt"))
writeLines(sample_ids, file.path(output_dir, "sample_ids.txt"))

write_f8 <- function(m, name) {
  con <- file(file.path(output_dir, name), "wb")
  on.exit(close(con))
  writeBin(as.numeric(m), con, size = 8, endian = "little")
}
write_f8(meth, "meth.f8")
write_f8(unmeth, "unmeth.f8")
write_f8(beta, "beta.f8")
write_f8(detection_p, "detection_p.f8")

writeLines(
  c(
    sprintf("preprocessMethod: %s", paste(preprocessMethod(Mset), collapse = " | ")),
    sprintf("minfi: %s", as.character(packageVersion("minfi"))),
    sprintf("n_probes: %d", length(probe_ids)),
    sprintf("n_samples: %d", length(sample_ids)),
    sprintf("created_at: %s", format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"))
  ),
  file.path(output_dir, "preprocess_metadata.txt")
)

log_msg("Done. Wrote outputs to %s", output_dir)
