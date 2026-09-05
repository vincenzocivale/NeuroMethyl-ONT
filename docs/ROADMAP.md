# Research roadmap

## V0 — Information-preservation benchmark

Goal: isolate the value of continuous methylation and coverage at fixed loci.

- Train reference: GSE90496
- Model selection: GSE109379
- Primary real ONT evaluation: GSE289246
- Compare binary vs beta vs beta+coverage representations
- Reproduce crossNN / Sturgeon / MethyLYZR where feasible

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

## V3 — Multi-signal full-run CNS classification

Potential integration of methylation with CNV and diagnostically relevant sequence alterations.
