#!/usr/bin/env bash
# Re-run every cohort evaluation and write tables to $OUT (default outputs/results).
# Competitor predictions (Sturgeon/crossNN) are read from $COMP; produce them
# with scripts/competitors/ first. Requires NEUROMETHYL_DATA_ROOT.
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${NEUROMETHYL_DATA_ROOT:?set NEUROMETHYL_DATA_ROOT}"
D="$NEUROMETHYL_DATA_ROOT"
EXP=${EXP:-outputs/experiments}
COMP=${COMP:-outputs/benchmarks}
OUT=${OUT:-outputs/results}
REF="$D/derived/reference/GSE90496"
TAX="$D/derived/evaluation/magibu_v1/taxonomy/taxonomy_91_to_family.tsv"
Binary="$EXP/b0_fixed80_seed17_bs128_w16_opt/checkpoint_best.pt"
Beta="$EXP/c0_fixed80_seed17_bs128_w16_opt/checkpoint_best.pt"
Evidence="$EXP/e1_kappa2_fixed80_seed17_bs128_w2/checkpoint_best.pt"
RAPID="$D/derived/array_aligned/probe_level/RapidCNS2"
LABELS="$D/datasets/RapidCNS2/metadata/Supplementary_Tables_1_2.xlsx"
mkdir -p "$OUT"

python scripts/evaluation/evaluate_rapidcns2.py --reference-dir "$REF" \
  --binary-checkpoint "$Binary" --beta-checkpoint "$Beta" --evidence-checkpoint "$Evidence" --taxonomy "$TAX" \
  --rapid-dir "$RAPID/Rapid_CNS2" --rapid-labels-xlsx "$LABELS" \
  --sturgeon-rapid "$COMP/sturgeon/rapidcns2_predictions.tsv" --crossnn-rapid "$COMP/crossnn/rapidcns2_predictions.tsv" \
  --ont-dir "$RAPID/ONT_WGS" \
  --sturgeon-ont "$COMP/sturgeon/ont_wgs_predictions.tsv" --crossnn-ont "$COMP/crossnn/ont_wgs_predictions.tsv" \
  --output-dir "$OUT/rapidcns2" --perclass-output "$OUT/rapidcns2_perclass.tsv"

python scripts/evaluation/evaluate_rapidcns2_calibration.py --reference-dir "$REF" \
  --binary-checkpoint "$Binary" --rapid-dir "$RAPID/Rapid_CNS2" --rapid-labels-xlsx "$LABELS" \
  --output-dir "$OUT/rapidcns2_calibration"

python scripts/evaluation/evaluate_gse289246.py --raw-dir "$D/datasets/GSE289246/raw/extracted" \
  --soft-file "$D/datasets/GSE289246/metadata/GSE289246_family.soft.gz" --reference-dir "$REF" \
  --binary-checkpoint "$Binary" --beta-checkpoint "$Beta" --evidence-checkpoint "$Evidence" --taxonomy "$TAX" \
  --sturgeon-predictions "$COMP/sturgeon/gse289246_predictions.tsv" \
  --crossnn-predictions "$COMP/crossnn/gse289246_predictions.tsv" \
  --output "$OUT/gse289246/gse289246_predictions.tsv"

python scripts/evaluation/evaluate_gse109379.py \
  --mnp-dir "$D/datasets/GSE109379/processed/mnp_reconstruction" \
  --manifest "$D/datasets/GSE109379/processed/idat_sample_manifest.tsv" --reference-dir "$REF" \
  --binary-checkpoint "$Binary" --beta-checkpoint "$Beta" --taxonomy "$TAX" \
  --sturgeon-predictions "$COMP/sturgeon/gse109379_predictions.tsv" \
  --crossnn-predictions "$COMP/crossnn/gse109379_predictions.tsv" --output-dir "$OUT/gse109379"

python scripts/evaluation/evaluate_gse209865.py --raw-dir "$D/datasets/GSE209865/raw/extracted" \
  --soft-file "$D/datasets/GSE209865/metadata/GSE209865_family.soft.gz" --reference-dir "$REF" \
  --binary-checkpoint "$Binary" --taxonomy "$TAX" \
  --sturgeon-predictions "$COMP/sturgeon/gse209865_predictions.tsv" \
  --crossnn-predictions "$COMP/crossnn/gse209865_predictions.tsv" --output-dir "$OUT/gse209865"

python scripts/evaluation/evaluate_magibu_pools.py --magibu-root "$D/derived/evaluation/magibu_v1" \
  --binary-checkpoint "$Binary" --beta-checkpoint "$Beta" --taxonomy "$TAX" --reference-dir "$REF" \
  --sturgeon-predictions "$COMP/sturgeon/magibu_v1_beta_only_predictions.tsv" \
  --crossnn-predictions "$COMP/crossnn/magibu_v1_beta_only_predictions.tsv" \
  --output-dir "$OUT/magibu_pool_comparison"

python scripts/evaluation/evaluate_our_samples.py \
  --counts-dir "$D/derived/evaluation/our_samples_v1/counts/low_coverage" \
  --binary-checkpoint "$Binary" --beta-checkpoint "$Beta" --reference-dir "$REF" --taxonomy "$TAX" \
  --coverage-label "low coverage (~10min ONT)" \
  --crossnn-predictions "$COMP/crossnn/our_samples_low_coverage_predictions.tsv" \
  --output "$OUT/our_samples/mb_low_coverage_predictions.tsv"
