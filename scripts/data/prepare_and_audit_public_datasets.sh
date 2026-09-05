#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /path/to/CNSNanoporeData"
  exit 1
fi

DATA_ROOT="$(readlink -f "$1")"
REGISTRY="$DATA_ROOT/registry"
AUDIT="$REGISTRY/audit"

mkdir -p "$AUDIT"

log() {
  printf "\n[%s] %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

extract_tar_once() {
  local archive="$1"
  local outdir="$2"

  mkdir -p "$outdir"
  if [[ -f "$outdir/.extraction_complete" ]]; then
    echo "Already extracted: $archive"
    return
  fi

  log "Extracting $(basename "$archive") -> $outdir"
  tar -xf "$archive" -C "$outdir"
  touch "$outdir/.extraction_complete"
}

extract_zip_once() {
  local archive="$1"
  local outdir="$2"

  mkdir -p "$outdir"
  if [[ -f "$outdir/.extraction_complete" ]]; then
    echo "Already extracted: $archive"
    return
  fi

  log "Extracting $(basename "$archive") -> $outdir"
  unzip -q "$archive" -d "$outdir"
  touch "$outdir/.extraction_complete"
}

log "Checking expected core files"

expected=(
  "$DATA_ROOT/datasets/GSE90496/processed/beta/GSE90496_beta.txt.gz"
  "$DATA_ROOT/datasets/GSE90496/metadata/GSE90496_methylationclassdescription.xlsx"
  "$DATA_ROOT/datasets/GSE90496/metadata/GSE90496_family.soft.gz"
  "$DATA_ROOT/datasets/GSE109379/processed/GSE109379_processed_data.txt.gz"
  "$DATA_ROOT/datasets/GSE109379/metadata/GSE109379_methylationclassdescription.xlsx"
  "$DATA_ROOT/datasets/GSE109379/metadata/GSE109379_family.soft.gz"
  "$DATA_ROOT/datasets/GSE289246/raw/bedmethyl/GSE289246_RAW.tar"
  "$DATA_ROOT/datasets/GSE289246/metadata/GSE289246_family.soft.gz"
  "$DATA_ROOT/datasets/GSE209865/raw/binary_calls/GSE209865_RAW.tar"
  "$DATA_ROOT/datasets/GSE209865/processed/GSE209865_methylation_merged_samples.tsv.gz"
  "$DATA_ROOT/datasets/GSE209865/metadata/GSE209865_family.soft.gz"
  "$DATA_ROOT/datasets/RapidCNS2/raw/ONT_WGS/ONT_WGS.zip"
  "$DATA_ROOT/datasets/RapidCNS2/raw/Rapid_CNS2/Rapid-CNS2.zip"
)

missing=0
for f in "${expected[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "MISSING: $f"
    missing=1
  fi
done

if [[ "$missing" -ne 0 ]]; then
  echo "Some expected files are missing. Aborting extraction."
  exit 2
fi

log "Recording archive listings before extraction"

tar -tf "$DATA_ROOT/datasets/GSE289246/raw/bedmethyl/GSE289246_RAW.tar" \
  > "$AUDIT/GSE289246_archive_contents.txt"

tar -tf "$DATA_ROOT/datasets/GSE209865/raw/binary_calls/GSE209865_RAW.tar" \
  > "$AUDIT/GSE209865_archive_contents.txt"

unzip -Z1 "$DATA_ROOT/datasets/RapidCNS2/raw/ONT_WGS/ONT_WGS.zip" \
  > "$AUDIT/RapidCNS2_ONT_WGS_archive_contents.txt"

unzip -Z1 "$DATA_ROOT/datasets/RapidCNS2/raw/Rapid_CNS2/Rapid-CNS2.zip" \
  > "$AUDIT/RapidCNS2_archive_contents.txt"

log "Extracting archives while preserving original downloads"

extract_tar_once \
  "$DATA_ROOT/datasets/GSE289246/raw/bedmethyl/GSE289246_RAW.tar" \
  "$DATA_ROOT/datasets/GSE289246/raw/extracted"

extract_tar_once \
  "$DATA_ROOT/datasets/GSE209865/raw/binary_calls/GSE209865_RAW.tar" \
  "$DATA_ROOT/datasets/GSE209865/raw/extracted"

extract_zip_once \
  "$DATA_ROOT/datasets/RapidCNS2/raw/ONT_WGS/ONT_WGS.zip" \
  "$DATA_ROOT/datasets/RapidCNS2/raw/ONT_WGS/extracted"

extract_zip_once \
  "$DATA_ROOT/datasets/RapidCNS2/raw/Rapid_CNS2/Rapid-CNS2.zip" \
  "$DATA_ROOT/datasets/RapidCNS2/raw/Rapid_CNS2/extracted"

log "Building post-extraction inventory"

find "$DATA_ROOT/datasets" \
  -type f \
  ! -name '.extraction_complete' \
  -printf '%p\t%s\n' \
  | sort > "$REGISTRY/files_inventory_postextract.tsv"

du -sh "$DATA_ROOT"/datasets/* \
  > "$AUDIT/dataset_sizes_postextract.txt"

log "Collecting file-extension counts"

python - "$DATA_ROOT" "$AUDIT" <<'PY'
from pathlib import Path
from collections import Counter
import sys

data_root = Path(sys.argv[1])
audit = Path(sys.argv[2])

rows = []
for ds in sorted((data_root / "datasets").iterdir()):
    if not ds.is_dir():
        continue
    c = Counter()
    n = 0
    for p in ds.rglob("*"):
        if p.is_file() and p.name != ".extraction_complete":
            n += 1
            name = p.name.lower()
            if name.endswith(".bed.gz"):
                ext = ".bed.gz"
            elif name.endswith(".tsv.gz"):
                ext = ".tsv.gz"
            elif name.endswith(".txt.gz"):
                ext = ".txt.gz"
            elif name.endswith(".soft.gz"):
                ext = ".soft.gz"
            else:
                ext = p.suffix.lower() or "<no_ext>"
            c[ext] += 1
    for ext, count in sorted(c.items()):
        rows.append((ds.name, ext, count))

with (audit / "file_type_counts.tsv").open("w") as f:
    f.write("dataset_id\tfile_type\tcount\n")
    for row in rows:
        f.write("\t".join(map(str, row)) + "\n")
PY

log "Writing first-line/header samples for text-like files"

python - "$DATA_ROOT" "$AUDIT" <<'PY'
from pathlib import Path
import gzip
import sys

data_root = Path(sys.argv[1])
audit = Path(sys.argv[2])

candidate_suffixes = (
    ".bed", ".bed.gz", ".tsv", ".tsv.gz", ".txt", ".txt.gz", ".csv", ".csv.gz"
)

out = audit / "text_file_headers.tsv"
with out.open("w") as w:
    w.write("dataset_id\tpath\theader_preview\n")
    for ds in sorted((data_root / "datasets").iterdir()):
        if not ds.is_dir():
            continue
        for p in ds.rglob("*"):
            if not p.is_file():
                continue
            low = p.name.lower()
            if not any(low.endswith(s) for s in candidate_suffixes):
                continue
            try:
                opener = gzip.open if low.endswith(".gz") else open
                with opener(p, "rt", errors="replace") as f:
                    line = f.readline().rstrip("\n\r")
                line = line.replace("\t", "\\t")[:1000]
                w.write(f"{ds.name}\t{p}\t{line}\n")
            except Exception as e:
                w.write(f"{ds.name}\t{p}\t<ERROR: {e}>\n")
PY

log "Done"

echo
echo "Audit outputs:"
echo "  $AUDIT/dataset_sizes_postextract.txt"
echo "  $AUDIT/file_type_counts.tsv"
echo "  $AUDIT/text_file_headers.tsv"
echo "  $REGISTRY/files_inventory_postextract.tsv"
echo
echo "Next recommended commands:"
echo "  column -t -s \$'\\t' '$AUDIT/file_type_counts.tsv' | less -S"
echo "  sed -n '1,80p' '$AUDIT/text_file_headers.tsv'"
