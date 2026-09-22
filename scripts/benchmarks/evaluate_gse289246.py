#!/usr/bin/env python3
"""B0 vs C0 vs EB1 vs E1 vs Sturgeon vs crossNN on GSE289246 -- this project's own
registry (registry/datasets.tsv) calls this the *primary* ONT test cohort,
but no model in this project had ever been run on it until now (only
Sturgeon had, via a bed-only path that never went through counts). Real
per-CpG coverage, not pooled, not synthetic.

Scope note: GSE289246 has 139 samples in its GEO metadata (rich class
diversity -- 35 diagnoses), but only the first 22 (all "Glioblastoma,
IDH-wildtype") have their raw bedMethyl actually downloaded under
datasets/GSE289246/raw/extracted/ in this environment. This script can only
score those 22 -- a real, if narrow, gap-closer (GBM recall only), not the
full cohort. Ground truth comes directly from the GEO family.soft.gz
"tumor type:" characteristic, not inferred.
"""

from __future__ import annotations

import argparse
import gzip
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from compare_rapidcns2_models import EvidenceBayesCheckpoint, load_torch_checkpoint  # noqa: E402

from neuromethyl_ont.data.array_reference import load_feature_index  # noqa: E402
from neuromethyl_ont.data.canonical import (  # noqa: E402
    aggregate_to_probe_level,
    to_coordinate_level,
)
from neuromethyl_ont.data.gse289246 import read_gse289246_sample  # noqa: E402
from neuromethyl_ont.evaluation.classification import softmax_probs  # noqa: E402
from neuromethyl_ont.evaluation.family_taxonomy import (  # noqa: E402
    aggregate_to_family,
    sturgeon_model_class,
)
from neuromethyl_ont.models.evidence_bayes import predict_log_joint_torch  # noqa: E402
from neuromethyl_ont.training.trainer import predict_logits  # noqa: E402

# Only the diagnosis actually present in the 22 locally-downloaded samples.
# Extend if more raw bedMethyl files are ever downloaded for this cohort.
GEO_DIAGNOSIS_TO_FAMILY = {
    "Glioblastoma, IDH-wildtype": "GBM",
}


def parse_soft_labels(soft_path: Path) -> dict[str, str]:
    """GSM accession -> raw 'tumor type:' string, from the GEO family SOFT file."""
    labels: dict[str, str] = {}
    gsm = None
    with gzip.open(soft_path, "rt") as fh:
        for line in fh:
            m = re.match(r"^!Sample_geo_accession = (\S+)", line)
            if m:
                gsm = m.group(1)
                continue
            m = re.match(r"^!Sample_characteristics_ch1 = tumor type: (.+)$", line)
            if m and gsm:
                labels[gsm] = m.group(1).strip()
    return labels


def reindex(probs: np.ndarray, source: list[str], target: list[str]) -> np.ndarray:
    out = np.zeros((len(probs), len(target)), dtype=np.float64)
    for col, family in enumerate(source):
        out[:, target.index(family)] = probs[:, col]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--raw-dir", type=Path, required=True, help="datasets/GSE289246/raw/extracted"
    )
    ap.add_argument("--soft-file", type=Path, required=True)
    ap.add_argument("--reference-dir", type=Path, required=True)
    ap.add_argument("--b0-checkpoint", type=Path, required=True)
    ap.add_argument("--c0-checkpoint", type=Path, required=True)
    ap.add_argument("--eb1-checkpoint-dir", type=Path, required=True)
    ap.add_argument(
        "--e1-checkpoint", type=Path, default=None, help="optional; omit to skip E1 entirely"
    )
    ap.add_argument("--e1-kappa", type=float, default=2.0)
    ap.add_argument("--taxonomy", type=Path, required=True)
    ap.add_argument("--sturgeon-predictions", type=Path, required=True)
    ap.add_argument("--crossnn-predictions", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    feature_index = load_feature_index(args.reference_dir / "feature_index.json")
    probe_pos = {probe: i for i, probe in enumerate(feature_index.probe_ids)}
    mu = (
        pd.read_parquet(args.reference_dir / "probe_statistics.parquet")
        .set_index("probe_id")["mean_beta"]
        .reindex(feature_index.probe_ids)
        .to_numpy()
    )

    taxonomy = pd.read_csv(args.taxonomy, sep="\t")
    family_by_class = {
        str(row.model_class): str(row.family_code).strip()
        for row in taxonomy.itertuples(index=False)
        if pd.notna(row.family_code) and str(row.family_code).strip()
    }

    soft_labels = parse_soft_labels(args.soft_file)
    files = sorted(args.raw_dir.glob("*.CG.bed.gz"))
    if not files:
        ap.error(f"no *.CG.bed.gz under {args.raw_dir}")

    n = feature_index.n_features
    # Row index into m/u is the count of *included* samples so far, not the
    # file-enumeration index -- a skipped sample must never leave a blank
    # row in the middle of the array (caught before this ever ran: with a
    # skip, m[:len(sample_ids)] would silently keep a zero row from a
    # skipped file instead of the next included file's data).
    m_rows, u_rows = [], []
    sample_ids, true_families, skipped = [], [], []
    for path in files:
        # Sturgeon/crossNN predictions (built from the same raw files by
        # build_sturgeon_bed_inputs.py) key on this exact id -- path.stem
        # alone leaves the ".CG.bed" suffix on (double extension), which
        # silently zero-matched every sample against those tables the first
        # time this ran (n=0 for both, caught by checking the printed
        # per-model rates rather than trusting a clean exit).
        sample_id = path.name.removesuffix(".CG.bed.gz")
        gsm = sample_id.split("_")[0]
        diagnosis = soft_labels.get(gsm)
        family = GEO_DIAGNOSIS_TO_FAMILY.get(diagnosis or "")
        if family is None:
            skipped.append((gsm, diagnosis))
            continue
        raw = read_gse289246_sample(path)
        coord = to_coordinate_level(raw, sample_id=sample_id)
        probe = aggregate_to_probe_level(coord)
        probe = probe.dropna(subset=["methylated_count", "unmethylated_count"])
        pos = probe["probe_id"].map(probe_pos)
        valid = pos.notna()
        idx = pos.loc[valid].astype(np.int64).to_numpy()
        m_row = np.zeros(n, dtype=np.float64)
        u_row = np.zeros(n, dtype=np.float64)
        m_row[idx] = probe.loc[valid, "methylated_count"].to_numpy(dtype=np.float64)
        u_row[idx] = probe.loc[valid, "unmethylated_count"].to_numpy(dtype=np.float64)
        m_rows.append(m_row)
        u_rows.append(u_row)
        sample_ids.append(sample_id)
        true_families.append(family)
        print(
            f"{sample_id}: {len(probe)} probes, diagnosis={diagnosis!r} -> family={family}",
            flush=True,
        )
    if skipped:
        print(f"WARNING: skipped {len(skipped)} sample(s) with unmapped diagnosis: {skipped}",
              file=sys.stderr)
    if not sample_ids:
        ap.error("no sample had a mappable diagnosis -- nothing to score")
    m, u = np.stack(m_rows), np.stack(u_rows)

    b0_model, b0_classes = load_torch_checkpoint(args.b0_checkpoint)
    c0_model, c0_classes = load_torch_checkpoint(args.c0_checkpoint)
    eb1 = EvidenceBayesCheckpoint(args.eb1_checkpoint_dir)

    from neuromethyl_ont.data.sequencing_simulator import (
        binary_transform,
        continuous_transform,
        evidence_transform,
    )

    b0_probs = softmax_probs(predict_logits(b0_model, binary_transform(m, u, threshold=0.6)))
    b0_fam, families = aggregate_to_family(b0_probs, b0_classes, family_by_class)
    c0_probs = softmax_probs(predict_logits(c0_model, continuous_transform(m, u, mu)))
    c0_fam, c0_families = aggregate_to_family(c0_probs, c0_classes, family_by_class)
    if c0_families != families:
        raise RuntimeError("B0/C0 family layouts disagree")
    eb1_probs = softmax_probs(predict_log_joint_torch(eb1, m, u, device=args.device))
    eb1_fam, eb1_families = aggregate_to_family(eb1_probs, eb1.classes, family_by_class)
    if eb1_families != families:
        raise RuntimeError("B0/EB1 family layouts disagree")

    e1_fam = None
    if args.e1_checkpoint is not None:
        e1_model, e1_classes = load_torch_checkpoint(args.e1_checkpoint)
        evidence = evidence_transform(m, u, mu, args.e1_kappa)
        e1_probs = softmax_probs(predict_logits(e1_model, evidence))
        e1_fam, e1_families = aggregate_to_family(e1_probs, e1_classes, family_by_class)
        if e1_families != families:
            raise RuntimeError("B0/E1 family layouts disagree")

    def external_probs(path, column_to_model_class=lambda c: c):
        table = pd.read_csv(path, sep="\t")
        score_columns = [c for c in table.columns if c not in {"sample_id", "number_probes"}]
        model_classes = [column_to_model_class(c) for c in score_columns]
        probs, ext_families = aggregate_to_family(
            table[score_columns].to_numpy(dtype=np.float64), model_classes, family_by_class
        )
        by_sample = {s: probs[i] for i, s in enumerate(table.sample_id)}
        return {s: reindex(by_sample[s][None, :], ext_families, families)[0] for s in by_sample}

    sturgeon_by_sample = external_probs(args.sturgeon_predictions, sturgeon_model_class)
    crossnn_by_sample = external_probs(args.crossnn_predictions)

    model_probs = {"B0": b0_fam, "C0": c0_fam, "EB1": eb1_fam}
    if e1_fam is not None:
        model_probs["E1"] = e1_fam
        model_probs["B0+E1 (avg)"] = 0.5 * b0_fam + 0.5 * e1_fam
    rows = []
    for i, sample_id in enumerate(sample_ids):
        true_idx = families.index(true_families[i])
        row = {"sample_id": sample_id, "true_family": true_families[i]}
        for name, probs in model_probs.items():
            row[f"{name}_predicted_family"] = families[probs[i].argmax()]
            row[f"{name}_true_family_probability"] = float(probs[i, true_idx])
        for name, table in [("Sturgeon", sturgeon_by_sample), ("crossNN", crossnn_by_sample)]:
            probs = table.get(sample_id)
            if probs is None:
                row[f"{name}_predicted_family"] = None
                row[f"{name}_true_family_probability"] = None
            else:
                row[f"{name}_predicted_family"] = families[probs.argmax()]
                row[f"{name}_true_family_probability"] = float(probs[true_idx])
        rows.append(row)

    result = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, sep="\t", index=False)

    report_models = ["B0", "C0", "EB1"]
    if e1_fam is not None:
        report_models += ["E1", "B0+E1 (avg)"]
    report_models += ["Sturgeon", "crossNN"]
    for name in report_models:
        col = f"{name}_predicted_family"
        acc = (result[col] == result.true_family).mean()
        print(f"{name}: top-1 correct-family rate = {acc:.3f} (n={result[col].notna().sum()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
