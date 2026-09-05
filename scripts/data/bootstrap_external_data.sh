#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-${NEUROMETHYL_DATA_ROOT:-}}"
if [[ -z "$DATA_ROOT" ]]; then
  echo "Usage: $0 /path/to/CNSNanoporeData"
  echo "or set NEUROMETHYL_DATA_ROOT"
  exit 1
fi

mkdir -p "$DATA_ROOT"/{registry,references/{hg38,chm13v2,probe_sets/{450k,epic,sturgeon,crossnn,mnpflex}},derived/{harmonized_450k,continuous_methylation,coverage_simulations,read_downsampling,genomic_regions,genomewide_features},benchmarks/{sturgeon,methylyzr,crossnn,mnpflex},cache}

for dataset in GSE90496 GSE109379 GSE289246 GSE209865 RapidCNS2 nanoDx_EGA; do
  mkdir -p "$DATA_ROOT/datasets/$dataset"/{raw,processed,metadata,splits}
done

mkdir -p "$DATA_ROOT/datasets/GSE90496/raw/idat"
mkdir -p "$DATA_ROOT/datasets/GSE90496/processed/beta"
mkdir -p "$DATA_ROOT/datasets/GSE289246/raw/bedmethyl"
mkdir -p "$DATA_ROOT/datasets/GSE209865/raw/binary_calls"
mkdir -p "$DATA_ROOT/datasets/RapidCNS2/raw"/{ONT_WGS,Rapid_CNS2,WGBS,methylation_panel}

echo "Initialized external data root: $DATA_ROOT"
