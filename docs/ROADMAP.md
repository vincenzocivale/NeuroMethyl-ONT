# Research roadmap

## V0 — Information-preservation benchmark

Goal: isolate the value of continuous methylation and coverage at fixed loci.

- Train reference: GSE90496
- Model selection: GSE109379
- Primary real ONT evaluation: GSE289246
- Compare binary vs beta vs beta+coverage representations
- Reproduce crossNN / Sturgeon / MethyLYZR where feasible

**Outcome**: B0 (binary representation) is the best-evidenced of the three
and is now the recommended production model — see
[`docs/B0_PRODUCTION_MODEL.md`](B0_PRODUCTION_MODEL.md). E1 (beta+coverage)
remains implemented but unvalidated: no training run or benchmark exists for
it yet, and its fixed global `kappa` is the same strategy that failed for
EB1 (`docs/EVIDENCE_BAYES_V1.md` §7). Reviving it is future work, not a
completed part of V0.

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
