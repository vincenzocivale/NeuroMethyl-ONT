# B0+E1 ensemble as the production classifier

## Decision

The recommended model for real classification (intra-operative and
end-of-run) is the **average of B0's and E1's softmax probabilities**
(`scripts/inference/classify_sample.py`), not either model alone. This
supersedes the earlier "B0 only" decision (2026-09-21) once E1 was actually
trained and a second real labelled cohort was checked.

## How the two individual models got here

**B0** (`BinaryLinear`, `configs/models/binary_linear.yaml`): thresholds
each probe's observed beta at 0.6 into a fixed +1/-1 call, independent of
coverage. Simple, and -- because the call only depends on *which side* of
0.6 a beta lands, never on how far -- structurally immune to a
per-probe scale/offset shift.

**E1** (`EvidenceLinearV1`, `configs/models/evidence_linear_v1.yaml`,
`kappa=2.0`, trained 2026-09-22 in
`outputs/experiments/e1_kappa2_fixed80_seed17_bs128_w2/`): coverage-shrunk
continuous feature, `posterior = (m + kappa*mu)/(n + kappa)`, feature =
`posterior - mu`. At low coverage this stays close to the training prior
(cautious); at high coverage it converges toward `beta - mu` (trusts the
data).

Both are `nn.Linear(n_features, 91)` -- identical architecture, only the
feature transform differs.

## Why neither alone is the best answer

Real-cohort results (`outputs/benchmarks/OURS_VS_COMPETITORS_BY_DATASET.md`,
`outputs/benchmarks/gse289246/gse289246_with_ensemble_predictions.tsv`,
`outputs/benchmarks/rapidcns2_model_comparison_with_e1/rapidcns2_model_comparison_metrics.tsv`):

| Cohort | B0 | E1 | **B0+E1 (avg)** | Sturgeon | crossNN |
|---|--:|--:|--:|--:|--:|
| Rapid-CNS2 (real coverage, median 2.9x, n=255) | 86.7% acc / 79.0% bal. acc / NLL 3.59 | 87.1% / 82.5% / 2.76 | **87.1% / 78.7% / NLL 2.53 (best)** | 72.2% acc | 87.5% acc / 72.7% bal. acc |
| GSE289246 (real coverage, ~1x per probe after aggregation -- see caveat below, n=18, GBM recall) | **100%** | 77.8% | **100%** | 94.4% | 100% |
| Magibu deep_pool (n=5) | 100% | -- | -- | 60% | 100% |
| Magibu lowpass_pool (~1x, n=16) | 100% | -- | -- | 81.3% | 100% |
| Own MB/ETMR samples (~10 min sequencing, n=8) | 100% | -- | -- | -- | 100% |

- On **GSE289246**, E1 alone drops to 77.8% -- it fails *confidently* on
  the same samples C0 does, always toward the same wrong class (LGNT), the
  signature of a platform/batch effect between GSE289246 (real ONT WGS)
  and the GSE90496 Illumina-array training reference. Continuous features
  (`beta - mu`, used by both C0 and E1) inherit that systematic offset
  directly into the logit; E1's shrinkage only partially attenuates it
  (fixes 7 of C0's 11 errors, not all). The **ensemble is not dragged down
  by this**: it still scores 100%, identical to B0 alone -- B0's
  confident-correct probability mass dominates the average on exactly the
  samples where E1 is confidently wrong.
- On **Rapid-CNS2**, the ensemble matches E1's accuracy gain over B0 and
  goes further on calibration (NLL 2.53, better than either model alone) --
  a real complementary-error benefit, not just "average of two similar
  numbers." The one cost: balanced accuracy is lower than E1 alone (78.7%
  vs 82.5%), since averaging dilutes E1's specific advantage on rare
  classes.

**Caveat, corrected 2026-09-22**: GSE289246 was initially assumed to be a
genuinely high-coverage ("end-of-run") cohort based on its docstring/registry
role ("primary ONT test cohort"). Checking the actual per-probe aggregated
counts shows mean coverage there is **~1x**, comparable to or lower than
Rapid-CNS2's ~2.9x -- so its failure mode is a **platform/array-vs-nanopore
batch effect, not a coverage-driven one**. A systematic `beta` offset
carries into the continuous feature regardless of how many reads support
it (bias, not variance), which is why E1's coverage-shrinkage doesn't fix
it. This means neither real cohort tested here is a confirmed genuinely
*high*-coverage labelled end-of-run case -- the "B0/ensemble wins at
end-of-run" conclusion currently rests on the fact that the one available
low-coverage batch-shifted cohort favors B0, plus the structural argument
(B0's threshold is offset-magnitude-invariant, E1's continuous feature is
not) rather than a directly confirmed high-coverage data point.

A per-sample **coverage-weighted** variant (trust E1 more at low coverage,
B0 more at high, tested 2026-09-22) was tried expecting it to help exactly
here -- it didn't (77.8%, identical to E1 alone), precisely because
GSE289246's low coverage told the weighting formula to trust E1, which is
wrong here for a platform reason unrelated to coverage. A **geometric-mean**
ensemble was also tried: better calibration on Rapid-CNS2 (NLL 2.43, best of
all variants) but worse on GSE289246 (94.4%, one dragged-down sample). The
flat 50/50 average remains the best overall choice across both real
cohorts and is what's implemented; the coverage-weighted variant should not
be used until a genuinely high-coverage labelled cohort is found to
validate it, and the batch-effect root cause (not coverage) is confirmed to
generalize.

## Why one model (pair) covers intra-operative and end-of-run

Both B0 and E1 encode each probe independently from raw
`(methylated_count, unmethylated_count)` evidence -- neither has a
depth-dependent branch in the model itself
(`src/neuromethyl_ont/data/sequencing_simulator.py`). Calling
`scripts/inference/classify_sample.py` on a partial intra-operative pileup
(few probes called) and calling it again on the finished run (nearly the
full panel called) uses the identical checkpoints and code path -- only how
much evidence is available at call time changes.

## Explainability

`src/neuromethyl_ont/evaluation/explainability.py` decomposes a linear
model's logit for any class into a per-probe contribution vector
(`contribution = weight[class, probe] * feature_value[probe]`), directly
from the model's own weight matrix -- no SHAP/attention/gradient machinery
needed, because both B0 and E1 *are* `Wx + b`. `explain_prediction` returns
the top-N contributing probes for the predicted class and its runner-ups;
`verify_decomposition` (used in `tests/test_explainability.py`) checks the
decomposition always sums back to the model's own forward-pass logit.

`scripts/inference/classify_sample.py` runs both checkpoints, reports the
ensemble's top-3 classification, and includes **both models'**
explainability reports side by side (`per_model.B0`/`per_model.E1`) -- since
they use different features, a merged single explanation would hide exactly
the disagreement pattern (e.g. the GSE289246 batch-effect case) that
justifies using the ensemble in the first place.

## What is not production

C0, EB1, and PerceiverNTv3V1 remain in the repo as validated ablation/
research baselines (directly cited by the existing benchmark reports and
design docs) but are not part of the production path. Neither B0 nor E1
alone is recommended once both checkpoints exist -- always run the ensemble
via `scripts/inference/classify_sample.py`.
