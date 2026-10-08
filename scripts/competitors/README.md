# Competitor adapters

`build_sturgeon_bed_*.py` convert each cohort to the BED inputs Sturgeon
expects; `run_sturgeon_predict.py`, `run_crossnn_predict.py` and
`run_methylyzr_predict.py` produce prediction tables that the scripts in
`scripts/evaluation/` consume; `merge_methylyzr_our_samples.py` merges
MethyLYZR predictions into the own-samples table.

Pin third-party versions and document preprocessing differences when adding a
competitor.
