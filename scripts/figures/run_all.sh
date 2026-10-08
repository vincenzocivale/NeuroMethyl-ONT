#!/usr/bin/env bash
# Regenerate every figure from the tables written by scripts/evaluation/run_all.sh.
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${NEUROMETHYL_DATA_ROOT:?set NEUROMETHYL_DATA_ROOT}"
D="$NEUROMETHYL_DATA_ROOT"
R=${OUT:-outputs/results}
F=${FIG:-$R/figures}
TAX="$D/derived/evaluation/magibu_v1/taxonomy/taxonomy_91_to_family.tsv"
LABELS="$D/datasets/RapidCNS2/metadata/Supplementary_Tables_1_2.xlsx"
MAG="$R/magibu_pool_comparison"
mkdir -p "$F"

python scripts/figures/plot_overall_accuracy_by_coverage.py \
  --magibu-metrics-tsv "$MAG/magibu_pool_model_comparison_metrics.tsv" \
  --rapidcns2-metrics-tsv "$R/rapidcns2/rapidcns2_model_comparison_metrics.tsv" --output-dir "$F"
python scripts/figures/plot_rapidcns2_coverage_and_confusion.py \
  --predictions-tsv "$R/rapidcns2/rapidcns2_low_coverage_predictions.tsv" \
  --taxonomy-tsv "$TAX" --rapid-labels-xlsx "$LABELS" --output-dir "$F"
python scripts/figures/plot_error_rate_by_class.py \
  --magibu-samples-tsv "$MAG/magibu_pool_model_comparison_samples.tsv" \
  --rapidcns2-perclass-tsv "$R/rapidcns2_perclass.tsv" --output-dir "$F"
python scripts/figures/plot_b0_per_class_heatmaps.py \
  --samples-tsv "$MAG/magibu_pool_model_comparison_samples.tsv" \
  --rapidcns2-perclass-tsv "$R/rapidcns2_perclass.tsv" --output-dir "$F"
python scripts/figures/plot_magibu_coverage_comparison.py \
  --samples-tsv "$MAG/magibu_pool_model_comparison_samples.tsv" --output-dir "$F"
python scripts/figures/plot_b0_calibration_top3.py --results-dir "$R/rapidcns2_calibration" \
  --calibration-file rapidcns2_b0_calibration.tsv \
  --top3-file rapidcns2_b0_top3.tsv \
  --class-metrics-file rapidcns2_class_metrics.tsv \
  --dataset-label "Rapid-CNS2" --output-prefix rapidcns2 --output-dir "$F"
python scripts/figures/plot_single_prediction_top3.py \
  --top3-file "$R/rapidcns2_calibration/rapidcns2_b0_top3.tsv" --output-dir "$F"
python scripts/figures/plot_our_samples_vs_competitors.py \
  --predictions-tsv "$R/our_samples/mb_low_coverage_predictions_with_methylyzr.tsv" \
  --output-png "$F/our_samples_heatmap.png"
