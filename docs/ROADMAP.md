# Research roadmap

## V0 — Information-preservation benchmark

Goal: isolate the value of continuous methylation and coverage at fixed loci.

- Train reference: GSE90496
- Model selection: GSE109379
- Primary real ONT evaluation: GSE289246
- Compare binary vs beta vs beta+coverage representations
- Reproduce crossNN / Sturgeon / MethyLYZR where feasible

**Outcome**: E1 (beta+coverage) was trained and evaluated 2026-09-22. It
beats B0 on Rapid-CNS2 (real coverage, median 2.9x) but loses badly on
GSE289246 (77.8% vs B0's 100%) due to a platform/array-vs-nanopore batch
effect in its continuous feature -- note GSE289246's actual per-probe
coverage turned out to be ~1x after aggregation, so this is a platform
effect, not specifically an end-of-run/high-coverage failure (no genuinely
high-coverage labelled cohort has confirmed the effect there yet).
Averaging B0's and E1's softmax probabilities matches or beats whichever
single model is better on every cohort tested, including calibration
(NLL) beyond both — this **B0+E1 ensemble** is now the recommended
production model, not B0 alone. See
[`docs/B0_PRODUCTION_MODEL.md`](B0_PRODUCTION_MODEL.md) for the
full evidence.

## V1 — Region-aware Nanopore model

Goal: use ONT CpGs surrounding array-informative loci/regions without requiring a fully genome-wide labelled ONT training cohort.

- genomic region representation
- variable numbers of CpGs per region
- region-level pooling
- breadth/depth controlled experiments

## V2 — Genome-wide Nanopore-native classifier

Goal: exploit variable-size genome-wide methylation observations directly.

Candidate modelling families:

- DeepSets
- attention pooling / Set Transformer
- hierarchical locus → region → patient encoders

`docs/PERCEIVER_NTV3_V1.md` is an early instance of this family: a
Perceiver-based set aggregator over NTv3 genomic-language-model CpG
embeddings x beta, run as its own pipeline in parallel with V0/V1 rather
than gated behind them. Its own "V1" in the filename numbers that pipeline's
internal iterations, not this roadmap's V0-V3 sequence -- it is currently
restricted to the array-anchored CpG panel (not yet genome-wide, see that
doc's #12) and E1 (the V0/V1 winner, once V0/V1 concludes) is its primary
baseline.

## V3 — Multi-signal full-run CNS classification

Potential integration of methylation with CNV and diagnostically relevant sequence alterations.
