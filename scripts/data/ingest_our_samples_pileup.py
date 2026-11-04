#!/usr/bin/env python3
"""Turn `modkit pileup` bedMethyl output for our own ETMR/MB samples into
counts aligned to the GSE90496 feature index -- the format Evidence needs
(methylated_count, unmethylated_count per probe), which the Magibu
beta-only pools and our own samples' existing entropy.tsv files cannot
provide.

Input: one bedMethyl file per sample (standard 18-column modkit pileup
output, produced with --include-bed pointed at the EPIC CpG annotation so
every row's (chrom, start, end) matches one EPIC probe exactly). Maps rows
back onto probe IDs via that same EPIC annotation BED (join on genomic
coordinates, not by re-deriving CpG context), then reindexes onto the
GSE90496 feature index order (missing probes -> m=u=0, i.e. no evidence,
the same convention `simulate_counts` uses).

Ground truth here is family-level only, and taken directly from the sample
naming convention (MB_* -> "MB", ETMR_* -> "ETMR") -- no per-subclass
molecular-subgroup label for these samples was found anywhere in the
source data (searched; see project memory). Never invent a subclass label.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from neuromethyl_ont.data.array_reference import load_feature_index  # noqa: E402

BEDMETHYL_COLUMNS = [
    "chrom", "start", "end", "mod_code", "score", "strand", "thick_start", "thick_end",
    "color", "valid_coverage", "fraction_modified", "n_mod", "n_canonical", "n_other_mod",
    "n_delete", "n_fail", "n_diff", "n_nocall",
]


def load_epic_coord_to_probe(epic_bed: Path) -> pd.Series:
    epic = pd.read_csv(
        epic_bed, sep="\t", header=None, usecols=[0, 1, 2, 3],
        names=["chrom", "start", "end", "probe_id"],
    )
    # Join on (chrom, start) only: the EPIC annotation spans the 2bp CpG
    # dinucleotide (start=C position, end=start+2), but modkit pileup (run
    # with --combine-strands) reports one 1bp row per position at the same
    # start -- the C position on the + strand -- so start alone identifies
    # the CpG unambiguously; requiring an exact end match as well silently
    # drops every row (verified: this was a real bug caught by 0 matches).
    epic["key"] = epic.chrom + ":" + epic.start.astype(str)
    n_dup = epic.key.duplicated().sum()
    if n_dup:
        # EPIC has a handful of design replicates at the same CpG (23/428201
        # here) -- keep the first probe_id deterministically, same
        # first-occurrence-wins policy build_array_reference uses for
        # duplicate probe IDs.
        print(f"WARNING: {epic_bed}: {n_dup} duplicate (chrom,start,end) keys, keeping first",
              file=sys.stderr)
        epic = epic.drop_duplicates(subset="key", keep="first")
    return epic.set_index("key")["probe_id"]


def family_from_sample_id(sample_id: str) -> str:
    # MB_591 is the one known exception: per the sample owner, it is the same tumor as one of the ETMR cases -- a
    # historical/naming artifact, not a true medulloblastoma. Both models
    # independently and confidently call it ETMR (~100%), which is exactly
    # what this override predicts and is why it was caught in the first
    # place -- never revert this back to the naive "MB_* -> MB" rule.
    if sample_id == "MB_591":
        return "ETMR"
    if sample_id.startswith("MB_"):
        return "MB"
    if sample_id.startswith("ETMR_"):
        return "ETMR"
    raise ValueError(f"no known family-from-name rule for sample_id={sample_id!r}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--bedmethyl-dir", type=Path, required=True, help="one *.bed per sample")
    ap.add_argument("--epic-bed", type=Path, required=True)
    ap.add_argument("--reference-dir", type=Path, required=True, help="for feature_index.json")
    ap.add_argument("--benchmark", required=True, help="e.g. our_samples_low_coverage")
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    feature_index = load_feature_index(args.reference_dir / "feature_index.json")
    probe_pos = {probe: i for i, probe in enumerate(feature_index.probe_ids)}
    coord_to_probe = load_epic_coord_to_probe(args.epic_bed)

    bed_files = sorted(args.bedmethyl_dir.glob("*.bed"))
    if not bed_files:
        ap.error(f"no .bed files under {args.bedmethyl_dir}")

    n_probes = feature_index.n_features
    m = np.zeros((len(bed_files), n_probes), dtype=np.float64)
    u = np.zeros((len(bed_files), n_probes), dtype=np.float64)
    manifest_rows = []
    for row, bed_path in enumerate(bed_files):
        sample_id = bed_path.stem
        table = pd.read_csv(bed_path, sep="\t", header=None, names=BEDMETHYL_COLUMNS)
        # --preset traditional already combines strands and restricts to CpG,
        # so each (chrom,start,end) should appear at most once per sample.
        if table.duplicated(subset=["chrom", "start"]).any():
            raise ValueError(f"{bed_path}: duplicate (chrom,start) rows")
        key = table.chrom + ":" + table.start.astype(str)
        probe_id = key.map(coord_to_probe)
        unmatched = probe_id.isna().sum()
        col = probe_id.map(probe_pos)
        valid = col.notna()
        idx = col.loc[valid].astype(np.int64).to_numpy()
        m[row, idx] = table.loc[valid, "n_mod"].to_numpy()
        u[row, idx] = table.loc[valid, "n_canonical"].to_numpy()
        family_code = family_from_sample_id(sample_id)
        manifest_rows.append(
            {
                "sample_id": sample_id,
                "benchmark": args.benchmark,
                "source_file": str(bed_path),
                "family_code": family_code,
                "n_bedmethyl_rows": len(table),
                "n_matched_to_epic": int(valid.sum()),
                "n_unmatched_to_epic": int(unmatched),
                "n_reads_total": int(table.n_mod.sum() + table.n_canonical.sum()),
            }
        )
        print(f"{sample_id}: {len(table)} rows, {int(valid.sum())} matched to feature index, "
              f"family={family_code}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "m.npy", m)
    np.save(args.output_dir / "u.npy", u)
    pd.DataFrame({"probe_id": feature_index.probe_ids}).to_csv(
        args.output_dir / "probe_ids.tsv", sep="\t", index=False
    )
    pd.DataFrame(manifest_rows).to_csv(args.output_dir / "manifest.tsv", sep="\t", index=False)
    print(f"Wrote {len(bed_files)} sample(s) to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
