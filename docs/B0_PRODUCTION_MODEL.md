# B0 as the production classifier

## Decision

**B0** (`BinaryLinear`, `configs/models/binary_linear.yaml`,
`src/neuromethyl_ont/models/evidence_linear.py`) is the recommended model for
real classification use, both intra-operative and end-of-run. This
supersedes any earlier framing of E1 as the eventual production model.

## Why B0, not E1

E1 (EvidenceLinearV1, `docs/EVIDENCE_LINEAR_V1.md`) is architecturally
appealing -- its coverage-shrinkage `kappa` parameter is meant to span
intra-operative (sparse evidence, shrink toward the training prior) and
end-of-run (dense evidence, trust the data) with a single model. But as of
this writing **E1 has never been trained or evaluated end-to-end**: no
checkpoint exists under `outputs/experiments/`, and neither
`outputs/benchmarks/OURS_VS_COMPETITORS_BY_DATASET.md` nor
`EB1_VS_COMPETITORS.md` report a single real accuracy number for it.
`docs/EVIDENCE_LINEAR_V1.md` §22 itself frames the whole milestone as
"contingent on EvidenceLinearV1 actually showing the effect." Its `kappa=2.0`
is a fixed, unswept hyperparameter, and the same global-kappa strategy was
tried on the closely related EB1 model and failed empirically -- accuracy
fell from 95.5% to 75.7% as simulated depth increased
(`docs/EVIDENCE_BAYES_V1.md` §7). Reviving E1 is real future work: it needs
an actual training run plus a `kappa` sweep, not just re-labeling.

B0 has concrete, validated results (`outputs/benchmarks/OURS_VS_COMPETITORS_BY_DATASET.md`):

| Cohort | B0 | Sturgeon | crossNN |
|---|--:|--:|--:|
| Rapid-CNS2 (real coverage, median 2.9x, n=255) | 86.7% acc / 79.0% balanced acc | 72.2% acc | 87.5% acc / 72.7% balanced acc |
| Magibu deep_pool (n=5) | 100% | 60% | 100% |
| Magibu lowpass_pool (~1x, n=16) | 100% | 81.3% | 100% |
| Own MB/ETMR samples (~10 min sequencing, n=8) | 100% | -- | 100% |

B0 beats Sturgeon everywhere and matches or beats crossNN on balanced
accuracy across the board, with the simplest architecture available
(`nn.Linear(n_features, 91)` on a fixed 0.6 binary threshold, zero shrinkage
hyperparameters).

## Why one model covers intra-operative and end-of-run

B0 encodes each probe independently from raw `(methylated_count,
unmethylated_count)` evidence via `binary_transform` -- a probe with `n = 0`
observed reads simply contributes 0 (unobserved), any other probe is called
+1/-1 by comparing its observed beta to the 0.6 threshold
(`src/neuromethyl_ont/data/sequencing_simulator.py`). There is no
depth-dependent branch in the model or the encoding. Calling
`scripts/inference/classify_sample.py` on a partial intra-operative pileup
(few probes called) and calling it again on the finished run (nearly the
full panel called) uses the identical model, checkpoint, and code path --
the only difference is how much evidence is available at call time.

## Explainability

`src/neuromethyl_ont/evaluation/explainability.py` decomposes B0's linear
logit for any class into a per-probe contribution vector
(`contribution = weight[class, probe] * feature_value[probe]`), directly
from the model's own weight matrix -- no SHAP/attention/gradient machinery
needed, because the model *is* `Wx + b`. `explain_prediction` returns the
top-N contributing probes for the predicted class and its runner-ups;
`verify_decomposition` (used in `tests/test_explainability.py`) checks the
decomposition always sums back to the model's own forward-pass logit.

`scripts/evaluation/explain_sample.py` and `scripts/inference/classify_sample.py`
both call into this module -- the latter bundles top-3 classification with
its explainability report in one JSON output, intended as the single entry
point for both intra-op and end-of-run use.

## What is not production

C0, EB1, and PerceiverNTv3V1 remain in the repo as validated ablation/
research baselines (they are directly cited by the existing benchmark
reports and design docs) but are not the recommended model for real
classification. E1 remains implemented but explicitly unvalidated; do not
treat its config or docstrings as evidence it has been shown to work.
