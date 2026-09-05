#!/usr/bin/env bash
set -euo pipefail

# NeuroMethyl-ONT public dataset downloader
# Usage:
#   bash download_neuromethyl_datasets.sh /path/to/CNSNanoporeData core
#   bash download_neuromethyl_datasets.sh /path/to/CNSNanoporeData arrays-raw
#   bash download_neuromethyl_datasets.sh /path/to/CNSNanoporeData extras
#   bash download_neuromethyl_datasets.sh /path/to/CNSNanoporeData all-public
#
# Modes:
#   core       : data needed for the first experiments (~26 GB compressed)
#   arrays-raw : raw IDAT archives for GSE90496/GSE109379 (~32 GB compressed)
#   extras     : Rapid-CNS2 WGBS + methylation panel (~1.1 GB compressed)
#   all-public : core + arrays-raw + extras
#
# Controlled EGA data are intentionally NOT downloaded here.

DATA_ROOT="${1:-}"
MODE="${2:-core}"

if [[ -z "$DATA_ROOT" ]]; then
  echo "ERROR: provide DATA_ROOT as first argument." >&2
  echo "Example: bash $0 /raid/DATASETS/CNSNanoporeData core" >&2
  exit 2
fi

case "$MODE" in
  core|arrays-raw|extras|all-public) ;;
  *)
    echo "ERROR: unknown mode '$MODE'. Use core, arrays-raw, extras, or all-public." >&2
    exit 2
    ;;
esac

if ! command -v wget >/dev/null 2>&1; then
  echo "ERROR: wget is required." >&2
  exit 2
fi

mkdir -p "$DATA_ROOT"/{registry,references,datasets,derived,benchmarks,cache}
LOG_DIR="$DATA_ROOT/registry/download_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/download_$(date +%Y%m%d_%H%M%S).log"
MANIFEST="$DATA_ROOT/registry/download_manifest.tsv"
CHECKSUMS="$DATA_ROOT/registry/source_md5.tsv"

exec > >(tee -a "$LOG_FILE") 2>&1

echo -e "dataset_id\tsource\tfile\tlocal_path" > /tmp/neuromethyl_manifest.$$
[[ -f "$MANIFEST" ]] || echo -e "dataset_id\tsource\tfile\tlocal_path" > "$MANIFEST"
[[ -f "$CHECKSUMS" ]] || echo -e "dataset_id\tfile\tmd5\tstatus" > "$CHECKSUMS"

fetch() {
  local dataset="$1"
  local url="$2"
  local out_dir="$3"
  local filename="$4"
  local expected_md5="${5:-}"

  mkdir -p "$out_dir"
  local out="$out_dir/$filename"

  echo
  echo "=================================================================="
  echo "[$dataset] $filename"
  echo "Target: $out"
  echo "=================================================================="

  wget \
    --continue \
    --tries=10 \
    --timeout=60 \
    --retry-connrefused \
    --progress=bar:force:noscroll \
    -O "$out" \
    "$url"

  local got_md5
  got_md5="$(md5sum "$out" | awk '{print $1}')"
  local status="recorded"

  if [[ -n "$expected_md5" ]]; then
    if [[ "$got_md5" == "$expected_md5" ]]; then
      status="verified"
      echo "MD5 verified: $got_md5"
    else
      status="FAILED"
      echo "ERROR: MD5 mismatch for $out" >&2
      echo " expected: $expected_md5" >&2
      echo " obtained: $got_md5" >&2
      exit 1
    fi
  else
    echo "MD5 recorded locally: $got_md5"
  fi

  echo -e "$dataset\t$filename\t$got_md5\t$status" >> "$CHECKSUMS"
  echo -e "$dataset\t$url\t$filename\t$out" >> "$MANIFEST"
}

geo_metadata() {
  local gse="$1"
  local bucket="$2"
  local out_dir="$DATA_ROOT/datasets/$gse/metadata"
  mkdir -p "$out_dir"

  # Family SOFT contains sample titles/characteristics and is useful for provenance.
  fetch "$gse" \
    "https://ftp.ncbi.nlm.nih.gov/geo/series/$bucket/$gse/soft/${gse}_family.soft.gz" \
    "$out_dir" "${gse}_family.soft.gz"
}

download_core() {
  echo "Downloading CORE dataset set (~26 GB compressed)."

  # ------------------------------------------------------------------
  # GSE90496: Capper/Heidelberg reference set; training data
  # We initially download processed beta values rather than the 22.7 GB IDAT tar.
  # ------------------------------------------------------------------
  mkdir -p "$DATA_ROOT/datasets/GSE90496"/{raw/idat,processed/beta,metadata,splits}
  fetch "GSE90496" \
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE90nnn/GSE90496/suppl/GSE90496_beta.txt.gz" \
    "$DATA_ROOT/datasets/GSE90496/processed/beta" \
    "GSE90496_beta.txt.gz"
  fetch "GSE90496" \
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE90nnn/GSE90496/suppl/GSE90496_methylationclassdescription.xlsx" \
    "$DATA_ROOT/datasets/GSE90496/metadata" \
    "GSE90496_methylationclassdescription.xlsx"
  geo_metadata "GSE90496" "GSE90nnn"

  # ------------------------------------------------------------------
  # GSE109379: independent Capper validation set
  # ------------------------------------------------------------------
  mkdir -p "$DATA_ROOT/datasets/GSE109379"/{raw/idat,processed,metadata,splits}
  fetch "GSE109379" \
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE109nnn/GSE109379/suppl/GSE109379_processed_data.txt.gz" \
    "$DATA_ROOT/datasets/GSE109379/processed" \
    "GSE109379_processed_data.txt.gz"
  fetch "GSE109379" \
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE109nnn/GSE109379/suppl/GSE109379_methylationclassdescription.xlsx" \
    "$DATA_ROOT/datasets/GSE109379/metadata" \
    "GSE109379_methylationclassdescription.xlsx"
  geo_metadata "GSE109379" "GSE109nnn"

  # ------------------------------------------------------------------
  # GSE289246: crossNN sequencing methylomes; BED/bedMethyl external test
  # ------------------------------------------------------------------
  mkdir -p "$DATA_ROOT/datasets/GSE289246"/{raw/bedmethyl,processed,metadata,splits}
  fetch "GSE289246" \
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE289nnn/GSE289246/suppl/GSE289246_RAW.tar" \
    "$DATA_ROOT/datasets/GSE289246/raw/bedmethyl" \
    "GSE289246_RAW.tar"
  geo_metadata "GSE289246" "GSE289nnn"

  # ------------------------------------------------------------------
  # GSE209865: legacy nanoDx/Sturgeon benchmark; BED/RData + merged calls
  # ------------------------------------------------------------------
  mkdir -p "$DATA_ROOT/datasets/GSE209865"/{raw/binary_calls,processed,metadata,splits}
  fetch "GSE209865" \
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE209nnn/GSE209865/suppl/GSE209865_RAW.tar" \
    "$DATA_ROOT/datasets/GSE209865/raw/binary_calls" \
    "GSE209865_RAW.tar"
  fetch "GSE209865" \
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE209nnn/GSE209865/suppl/GSE209865_methylation_merged_samples.tsv.gz" \
    "$DATA_ROOT/datasets/GSE209865/processed" \
    "GSE209865_methylation_merged_samples.tsv.gz"
  geo_metadata "GSE209865" "GSE209nnn"

  # ------------------------------------------------------------------
  # Rapid-CNS2 / MNP-Flex public processed validation data on Zenodo.
  # Only ONT-focused archives are part of core.
  # ------------------------------------------------------------------
  mkdir -p "$DATA_ROOT/datasets/RapidCNS2"/{raw/ONT_WGS,raw/Rapid_CNS2,raw/WGBS,raw/methylation_panel,processed,metadata,splits}
  fetch "RapidCNS2" \
    "https://zenodo.org/records/13351527/files/ONT_WGS.zip?download=1" \
    "$DATA_ROOT/datasets/RapidCNS2/raw/ONT_WGS" \
    "ONT_WGS.zip" \
    "f8058c153e23441077a715010d64f098"
  fetch "RapidCNS2" \
    "https://zenodo.org/records/13351527/files/Rapid-CNS2.zip?download=1" \
    "$DATA_ROOT/datasets/RapidCNS2/raw/Rapid_CNS2" \
    "Rapid-CNS2.zip" \
    "22043eed6d7f4c57ef93651fcddf4a38"
}

download_array_raw() {
  echo "Downloading raw array IDAT archives (~32 GB compressed)."
  mkdir -p "$DATA_ROOT/datasets/GSE90496/raw/idat" "$DATA_ROOT/datasets/GSE109379/raw/idat"

  fetch "GSE90496" \
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE90nnn/GSE90496/suppl/GSE90496_RAW.tar" \
    "$DATA_ROOT/datasets/GSE90496/raw/idat" \
    "GSE90496_RAW.tar"

  fetch "GSE109379" \
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE109nnn/GSE109379/suppl/GSE109379_RAW.tar" \
    "$DATA_ROOT/datasets/GSE109379/raw/idat" \
    "GSE109379_RAW.tar"
}

download_extras() {
  echo "Downloading optional Rapid-CNS2 non-ONT modalities (~1.1 GB compressed)."
  mkdir -p "$DATA_ROOT/datasets/RapidCNS2/raw/WGBS" "$DATA_ROOT/datasets/RapidCNS2/raw/methylation_panel"

  fetch "RapidCNS2" \
    "https://zenodo.org/records/13351527/files/WGBS.zip?download=1" \
    "$DATA_ROOT/datasets/RapidCNS2/raw/WGBS" \
    "WGBS.zip" \
    "6396a67c9f39cea42c1dd8e9bbbad497"

  fetch "RapidCNS2" \
    "https://zenodo.org/records/13351527/files/Methylation%20panel.zip?download=1" \
    "$DATA_ROOT/datasets/RapidCNS2/raw/methylation_panel" \
    "Methylation_panel.zip" \
    "e67f71093266b4c861b41d425ec39f9b"
}

case "$MODE" in
  core)
    download_core
    ;;
  arrays-raw)
    download_array_raw
    ;;
  extras)
    download_extras
    ;;
  all-public)
    download_core
    download_array_raw
    download_extras
    ;;
esac

# Remove exact duplicate lines introduced by resumed/repeated runs while preserving header.
python3 - "$MANIFEST" "$CHECKSUMS" <<'PY'
import sys
for path in sys.argv[1:]:
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    if not lines:
        continue
    header, rest = lines[0], lines[1:]
    seen = set()
    out = []
    for line in rest:
        if line not in seen:
            seen.add(line)
            out.append(line)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(header)
        f.writelines(out)
PY

echo
echo "Download mode '$MODE' completed."
echo "Log:       $LOG_FILE"
echo "Manifest:  $MANIFEST"
echo "Checksums: $CHECKSUMS"
echo
echo "NOTE: EGA datasets are controlled-access and are intentionally excluded."
