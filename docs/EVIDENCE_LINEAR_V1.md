# EvidenceLinearV1

Design spec for the V0 representation ablation (binary vs. continuous vs.
evidence-weighted methylation), implemented as `B0` / `C0` / `E1` below. This
document is the source of truth for `src/neuromethyl_ont/data/array_reference.py`,
`src/neuromethyl_ont/data/sequencing_simulator.py`,
`src/neuromethyl_ont/models/evidence_linear.py`, the `training/` and
`evaluation/` modules, and the configs under `configs/models/` and
`configs/experiments/v1_representation_ablation.yaml`. It refines
`docs/ROADMAP.md`'s "V0 — Information-preservation benchmark" milestone and
`docs/EXPERIMENT_POLICY.md`'s "Core V0 ablation" into an exact, testable
specification.

## 1. Goal

First experimental question:

> Does retaining continuous methylation and sequencing evidence improve CNS
> tumor classification compared with binary methylation representations?

Not tested yet in this milestone: genome-wide CpGs, Transformers, region
encoders, CNV, hierarchical classifiers, read-level models. This milestone
must first show that **the representation of evidence matters**, using the
smallest architecture that makes that claim credible.

## 2. Dataset roles

- **Train**: GSE90496 (n=2801, 91 methylation classes, 450K; the Heidelberg
  v11b4 reference cohort used by Sturgeon, crossNN, MethyLYZR). Not further
  split 80/20.
- **Validation**: GSE109379 (n=1104, Capper et al.'s prospective validation
  set). Used for model selection, early stopping, choosing kappa, and
  synthetic-depth validation.
- **Test (not used during development)**: GSE209865 (legacy Sturgeon/crossNN
  comparison), Rapid-CNS2, ONT_WGS, GSE289246. Rapid-CNS2/ONT-WGS are the
  primary coverage-aware benchmark.

## 3. Feature space

One canonical probe list built exclusively from GSE90496 training data (never
from a test set). Drop only: probes without sufficient valid values, probes
constant in training, duplicate probe IDs.

Saved layout (`$DATA_ROOT/derived/reference/GSE90496/`):

```text
beta.npy
sample_ids.tsv
probe_ids.tsv
labels.tsv
probe_statistics.parquet   # probe_id, mean_beta, std_beta, valid_fraction
feature_index.json
```

The same feature index is reused by every model (B0/C0/E1) and by validation
cohorts (aligned via `align_to_feature_index`).

## 4. Three representations, one architecture

Same architecture, train set, feature set, optimizer, and schedule across all
three. Only the representation changes.

- **B0 -- BinaryLinear**: `beta_hat > 0.6 -> +1`, `<= 0.6 -> -1`, missing
  (`n == 0`) -> `0`. Conceptually close to crossNN/Sturgeon.
- **C0 -- ContinuousLinear**: `x_i = beta_hat_i - mu_i`, where `mu_i` is the
  training-set mean beta for probe `i`. Missing -> `0` (i.e. "no evidence
  relative to the probe's background", not "unmethylated").
- **E1 -- EvidenceLinearV1**: given ONT counts `m_i` (methylated reads),
  `u_i` (unmethylated reads), `n_i = m_i + u_i`, and prior `mu_i`:

  ```text
  posterior_beta_i = (m_i + kappa * mu_i) / (n_i + kappa)
  x_i = posterior_beta_i - mu_i
      = n_i / (n_i + kappa) * (beta_hat_i - mu_i)
  ```

  `n_i = 0 -> x_i = 0`. High coverage -> `x_i ~ beta_hat_i - mu_i`. Low
  coverage shrinks `x_i` toward zero. Coverage is used exclusively as
  **confidence in the methylation evidence**, never as an independent
  feature -- otherwise the model could learn `sequencing depth -> tumor
  class` from batch/platform artifacts instead of biology.

## 5. Architecture

All three: `nn.Linear(n_features, 91, bias=True)`. No hidden layer, no
dropout, no attention, no Transformer. `logits = Wx + b`, `CrossEntropyLoss`.
If E1 beats B0/C0 under this constraint, the result cannot be attributed to
extra model capacity.

## 6. Training-time depth simulation

GSE90496 has no real coverage. Treat `beta_i` as the latent methylation
probability and simulate sequencing per sample/augmentation:

```python
n_i ~ Poisson(lambda)
m_i ~ Binomial(n_i, beta_i)
u_i = n_i - m_i
```

then apply the B0/C0/E1 transform above. Never compute features as
`beta * coverage` directly -- always sample counts first.

## 7. Depth (lambda) distribution

Generic distribution covering sparse to full-run sequencing, deliberately not
fit to Rapid-CNS2/ONT-WGS depth distributions (would bias toward the test
benchmark):

```text
lambda:      0.02  0.05  0.10 | 0.5  1  2  4 | 8  16  32 | inf (full array)
weight:      15% split evenly | 45% split evenly | 30% split evenly | 10%
```

`lambda = inf` means no sampling: `x_i = beta_i - mu_i` (C0/E1 coincide in
this limit) or `beta_i > 0.6` (B0), directly from the array beta.

## 8. Binary augmentation

B0 sees the same `n_i` availability pattern as C0/E1. `n_i == 0 -> x_i = 0`;
otherwise threshold `beta_hat_i = m_i/n_i` at `0.6`. Increasing `n_i` never
changes the binary value once evidence exists -- by construction, 1/1
methylated reads and 30/30 methylated reads give the identical binary
feature. That collapse is exactly the limitation E1 is designed to fix.

## 9. Continuous augmentation

`x_i = beta_hat_i - mu_i` for `n_i > 0`, else `0`.

## 10. Kappa

Configurable; first sweep `{0, 1, 2, 4, 8}`. `kappa = 0` removes
coverage-aware shrinkage and E1 degenerates toward C0 -- a built-in ablation.
First run: `kappa = 2`, no sweep yet.

## 11. Training config (first run)

```yaml
model: evidence_linear_v1
optimizer: {name: AdamW, lr: 3.0e-4, weight_decay: 1.0e-4}
training: {batch_size: 16, max_epochs: 80, early_stopping_patience: 12, mixed_precision: bf16}
loss: {name: cross_entropy}
seed: 17
```

No LR scheduler, no multi-seed runs yet.

## 12. Class imbalance

Record the 91-class distribution. First run: plain cross-entropy, no class
weighting. Only add `1/sqrt(class_frequency)` weighting later if macro-F1
shows a clear problem on rare classes.

## 13. Validation protocol (GSE109379)

- **Full-array**: real beta values, to verify the taxonomy was learned.
- **Synthetic sequencing**: FIXED realizations at
  `lambda in {0.02, 0.1, 0.5, 1, 2, 4, 8, 16, 32, full array}`, sampled once
  (same `(m, u, n)` draw at each depth reused across B0/C0/E1) and saved, so
  all three models are compared on identical synthetic evidence.

## 14. Metrics

Per depth: accuracy, balanced_accuracy, macro_F1, top3_accuracy, NLL, ECE,
plus `fraction(score >= 0.8)` / `accuracy | score >= 0.8` and the same at
`0.95` (Sturgeon's General-model operating thresholds).

## 15. Main V1 figure

x = sequencing depth, y = accuracy, three curves (B0/C0/E1). Hypothesis:
`B0 ~ C0 ~ E1` at very low depth; `C0 > B0` as depth grows;
`E1 >= C0 > B0` at high/full-run depth; the gap `E1 - B0` should grow with
available information.

## 16. External benchmarks (after model selection on GSE109379 only)

- **Legacy** (GSE209865, 415 ONT R9, binary calls): evaluate E1 at `n = 1`
  per observed CpG. Not expected to show the coverage-aware advantage; checks
  E1 is competitive in Sturgeon's original sparse-evidence setting.
- **Full-run** (Rapid-CNS2 n=300, ONT_WGS n=39): real `m, u, n, beta` from the
  canonical parquet outputs. This is where E1's advantage should show.

## 17. Taxonomy

Train on the original 91 GSE90496 classes, not the 87-class Sturgeon
taxonomy. A later `configs/taxonomies/sturgeon_general_87.yaml` will map
91 -> 87 class probabilities for apples-to-apples comparison; not built yet.

## 18. Repo layout

```text
src/neuromethyl_ont/
├── data/{array_reference.py, sequencing_simulator.py}
├── models/evidence_linear.py
├── training/{trainer.py, objectives.py}
└── evaluation/{classification.py, depth_curve.py}

scripts/data/build_array_reference.py
scripts/training/train_classifier.py
scripts/evaluation/evaluate_depth_curve.py

configs/models/{binary_linear.yaml, continuous_linear.yaml, evidence_linear_v1.yaml}
configs/experiments/v1_representation_ablation.yaml
```

## 19. Run outputs

```text
outputs/experiments/<run_id>/
├── config.yaml
├── checkpoint_best.pt
├── training_history.tsv
├── validation_metrics.json
├── depth_curve.tsv
├── per_class_metrics.tsv
└── metadata.json   # git commit, feature index checksum, datasets, n_features,
                     # n_classes, class mapping version
```

## 20. Required tests

1. Evidence transform reference value: `mu=0.5, m=8, u=2, n=10, kappa=2` ->
   `posterior=(8+2*0.5)/12=0.75` -> `x=0.25`.
2. Missing (`n=0`) -> `x=0`, never NaN, for B0/C0/E1.
3. High coverage: `n -> inf` => `x -> beta_hat - mu`.
4. Coverage effect: same `beta_hat=0.9`, `mu=0.5`; evidence magnitude at
   `n=1` is smaller than at `n=20`.
5. Simulator: `m >= 0`, `u >= 0`, `m + u = n`; `E[m/n] ~ beta` at sufficient
   coverage.

## 21. Success criteria

A. On GSE109379 full-array, E1/C0 are competitive with a good linear
   classifier. B. On the depth curve, `C0 > B0` as depth grows. C. `E1 >= C0`
   at medium/high depth. D. The E1 advantage grows toward full-run depth.
   E. No collapse vs. sparse-oriented methods on GSE209865. If A-E hold,
   move to V2.

## 22. Explicitly out of scope for this milestone

MLP, Transformer, Set Transformer, CpG embeddings, genomic position
embeddings, regional attention, CNV, sequence features, hierarchical loss,
multi-task learning, tumor purity branch. These are V2+ work, contingent on
EvidenceLinearV1 actually showing the effect above.
