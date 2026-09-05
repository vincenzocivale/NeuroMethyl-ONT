# External data layout

Large data lives outside Git. Point the repository to it through `NEUROMETHYL_DATA_ROOT`.

Recommended layout:

```text
CNSNanoporeData/
├── registry/
├── references/
│   ├── hg38/
│   ├── chm13v2/
│   └── probe_sets/
│       ├── 450k/
│       ├── epic/
│       ├── sturgeon/
│       ├── crossnn/
│       └── mnpflex/
├── datasets/
│   ├── GSE90496/{raw,processed,metadata,splits}/
│   ├── GSE109379/{raw,processed,metadata,splits}/
│   ├── GSE289246/{raw,processed,metadata,splits}/
│   ├── GSE209865/{raw,processed,metadata,splits}/
│   ├── RapidCNS2/{raw,processed,metadata,splits}/
│   └── nanoDx_EGA/{raw,processed,metadata,splits}/
├── derived/
│   ├── harmonized_450k/
│   ├── continuous_methylation/
│   ├── coverage_simulations/
│   ├── read_downsampling/
│   ├── genomic_regions/
│   ├── genomewide_features/
│   ├── ont_native/
│   │   └── coordinate_level/       # one Parquet per sample, one row per genomic observation
│   │       └── RapidCNS2/{ONT_WGS,Rapid_CNS2}/<sample_id>.parquet
│   └── array_aligned/
│       └── probe_level/            # one Parquet per sample, one row per probe_id (evidence-weighted)
│           └── RapidCNS2/{ONT_WGS,Rapid_CNS2}/<sample_id>.parquet
├── benchmarks/
│   ├── sturgeon/
│   ├── methylyzr/
│   ├── crossnn/
│   └── mnpflex/
└── cache/
```

## Raw / processed / derived

- `raw/`: immutable downloaded material or controlled-access material as received.
- `processed/`: dataset-specific standardized outputs produced directly from that dataset.
- `derived/`: cross-dataset or experiment-independent representations/harmonizations.
- `benchmarks/`: third-party model assets and reproducibility outputs.

## Provenance

Every processed/derived artifact should be reproducible from:

- source dataset and source files,
- sample/patient IDs,
- reference genome,
- preprocessing script/version,
- config,
- output schema.

Do not silently overwrite processed data when changing the reference genome, threshold, probe set, or caller.
