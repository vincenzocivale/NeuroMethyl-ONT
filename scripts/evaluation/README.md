# Evaluation scripts

One entry point per cohort; reusable metric logic lives in
`src/neuromethyl_ont/evaluation/`.

| Script | Cohort |
|---|---|
| `evaluate_rapidcns2.py` | Rapid-CNS2 (labelled) and ONT-WGS (agreement only) |
| `evaluate_rapidcns2_calibration.py` | Rapid-CNS2 reliability and top-3 |
| `evaluate_gse289246.py` | GSE289246 |
| `evaluate_gse109379.py` | GSE109379 (array, beta only) |
| `evaluate_gse209865.py` | GSE209865 (legacy binary calls) |
| `evaluate_magibu_pools.py` | Magibu pooled pseudo-samples (beta only) |
| `evaluate_our_samples.py` | own MB/ETMR samples |
| `evaluate_depth_curve.py` | synthetic depth curve on the GSE90496 holdout |
