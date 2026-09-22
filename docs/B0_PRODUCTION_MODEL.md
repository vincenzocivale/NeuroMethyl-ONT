# B0 as the production classifier

## Decision

**B0** (`BinaryLinear`, `configs/models/binary_linear.yaml`,
`src/neuromethyl_ont/models/evidence_linear.py`) is the recommended model for
real classification use, both intra-operative and end-of-run. This
supersedes any earlier framing of E1 as the eventual production model.

## Why B0, not E1

E1 (EvidenceLinearV1, `docs/EVIDENCE_LINEAR_V1.md`) was trained and evaluated
end-to-end on 2026-09-22 (`outputs/experiments/e1_kappa2_fixed80_seed17_bs128_w2/`,
`kappa=2.0`, same recipe as B0/C0) to settle whether its coverage-shrinkage
idea holds up. Result is cohort-dependent, and the cohort that matters most
(end-of-run, real high coverage) favors B0:

- **Rapid-CNS2** (real coverage, median 2.9x -- low/medium, closer to
  intra-operative): E1 beat B0/C0/EB1 on every metric -- 87.1% acc (B0:
  86.7%), 82.5% balanced acc (B0: 79.0%), NLL 2.76 (B0: 3.59).
- **GSE289246** (real per-CpG coverage, GBM recall, n=18 -- the primary
  real end-of-run-like labelled cohort): **B0 = 100%, E1 = 77.8%**, worse
  than even EB1 (83.3%). Root cause found by inspecting per-sample
  predictions: C0 and E1 both fail *confidently* on the same subset of
  samples, always toward the same wrong class (LGNT) -- the signature of a
  platform/batch effect between GSE289246 (real ONT WGS) and the GSE90496
  Illumina-array training reference. Continuous features (`beta - mu`, used
  by both C0 and E1) inherit that systematic offset directly into the
  logit; E1's shrinkage attenuates it (recovers 7 of C0's 11 errors) but
  doesn't eliminate it on samples where the offset is large enough to
  survive `kappa=2` shrinkage. B0's binary threshold is structurally immune
  to this: a shifted beta only flips the call if it crosses 0.6, so the
  *magnitude* of a batch-effect offset is irrelevant as long as it doesn't
  cross the threshold.

So the coverage-shrinkage idea is real (it clearly helps at low/medium
coverage) but is not free: it re-introduces a continuous-feature
vulnerability to platform batch effects that B0 doesn't have, and that
vulnerability does not shrink away with more coverage -- if anything, real
end-of-run ONT data pulls in more of exactly the continuous signal that can
carry the batch effect. Given the priority on end-of-run performance, this
tips the decision to B0.

B0 has concrete, validated results (`outputs/benchmarks/OURS_VS_COMPETITORS_BY_DATASET.md`,
`outputs/benchmarks/gse289246/gse289246_with_e1_predictions.tsv`):

| Cohort | B0 | E1 | Sturgeon | crossNN |
|---|--:|--:|--:|--:|
| Rapid-CNS2 (real coverage, median 2.9x, n=255) | 86.7% acc / 79.0% balanced acc | 87.1% acc / 82.5% balanced acc | 72.2% acc | 87.5% acc / 72.7% balanced acc |
| GSE289246 (real coverage, end-of-run-like, n=18, GBM recall) | 100% | 77.8% | 94.4% | 100% |
| Magibu deep_pool (n=5) | 100% | -- | 60% | 100% |
| Magibu lowpass_pool (~1x, n=16) | 100% | -- | 81.3% | 100% |
| Own MB/ETMR samples (~10 min sequencing, n=8) | 100% | -- | -- | 100% |

B0 beats Sturgeon everywhere and matches or beats crossNN on balanced
accuracy across the board, with the simplest architecture available
(`nn.Linear(n_features, 91)` on a fixed 0.6 binary threshold, zero shrinkage
hyperparameters) -- and, per the GSE289246 finding above, the one most
structurally robust to array-vs-nanopore platform shift.

E1 is not abandoned as a research direction (it is the better choice at
low/medium coverage, e.g. genuinely intra-operative use), but it is not the
production model given the end-of-run priority. Reviving it for that use
case would mean fixing the continuous-feature batch-effect vulnerability
first (e.g. per-cohort/per-platform recentering of `mu`, or a robustified
feature), not just re-running with a different `kappa`.

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

`scripts/inference/classify_sample.py` calls into this module -- it bundles
top-3 classification with its explainability report in one JSON output,
intended as the single entry point for both intra-op and end-of-run use.

## What is not production

C0, EB1, E1, and PerceiverNTv3V1 remain in the repo as validated ablation/
research baselines (they are directly cited by the existing benchmark
reports and design docs) but are not the recommended model for real
classification. E1 in particular is now validated (unlike the earlier
"unvalidated" framing) but loses to B0 specifically on the end-of-run-like
labelled cohort (GSE289246) -- see above.
