# Model names

The four model labels were renamed (approved 2026-11-04). Binary (formerly B0), Beta (formerly C0),
Evidence (formerly E1) and Consensus (formerly B0+E1) are the only names used in new text, tables and
figures.

| Old | New | Class / config | What it is |
|---|---|---|---|
| B0 | Binary | `BinaryLinear`, `configs/models/binary_linear.yaml` | linear model on the binary call, threshold 0.6 |
| C0 | Beta | `ContinuousLinear`, `configs/models/continuous_linear.yaml` | linear model on beta minus the training mean |
| E1 | Evidence | `EvidenceLinearV1`, `configs/models/evidence_linear_v1.yaml` | linear model on the coverage-shrunk posterior beta minus the mean |
| B0+E1 | Consensus | (ensemble, no class) | average of the Binary and Evidence softmax probabilities |

`EB1` (Beta-Binomial naive Bayes) keeps its name.

## Compatibility rule

- The mapping lives in one place: `LEGACY_MODEL_NAMES` in `src/neuromethyl_ont/model_names.py`.
- Anything that writes results (tables, JSON, figures) uses the new names.
- Anything that reads saved results passes them through `read_results_tsv` / `canonicalize_frame`
  (maps the `model`, `model_a`, `model_b` values and column labels such as `B0_confidence`) or
  `canonical_model_name`, so old and new files both work.
- Checkpoints store the representation (`binary`, `continuous`, `evidence`), not a model name, so old
  checkpoints load unchanged. Old CLI flags (`--b0-checkpoint`, `--c0-checkpoint`, `--e1-checkpoint`,
  `--e1-kappa`, `--alma-b0-checkpoint`, `--alma-e1-checkpoint`) still work as aliases of
  `--binary-checkpoint`, `--beta-checkpoint`, `--evidence-checkpoint`, `--evidence-kappa`,
  `--alma-binary-checkpoint`, `--alma-evidence-checkpoint`.
- Existing files under `outputs/` (experiment directories such as `b0_fixed80_...`, old TSVs) are never
  renamed or edited. Scripts and result files whose names contain `b0` (for example
  `plot_b0_*.py`, `rapidcns2_b0_calibration.tsv`) keep those names.
