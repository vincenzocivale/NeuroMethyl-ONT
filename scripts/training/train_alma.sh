#!/usr/bin/env bash
# Train B0/C0/E1 on the ALMA train split (same recipe as the CNS heads).
# Validation = ALMA val split (early stopping); test is never touched here.
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${NEUROMETHYL_DATA_ROOT:?set NEUROMETHYL_DATA_ROOT}"
D="$NEUROMETHYL_DATA_ROOT"
for spec in "b0:binary_linear" "c0:continuous_linear" "e1:evidence_linear_v1"; do
  id=${spec%%:*}; cfg=${spec#*:}
  python scripts/training/train_classifier.py \
    --model-config "configs/models/$cfg.yaml" \
    --reference-dir "$D/derived/reference/ALMA_discovery_train" \
    --validation-beta "$D/derived/evaluation/ALMA/val/beta.npy" \
    --validation-probe-ids "$D/derived/evaluation/ALMA/val/probe_ids.tsv" \
    --validation-labels "$D/derived/evaluation/ALMA/val/labels.tsv" \
    --validation-dataset ALMA_val --expected-classes 28 \
    --device cuda --run-id "alma_${id}_seed17"
done
