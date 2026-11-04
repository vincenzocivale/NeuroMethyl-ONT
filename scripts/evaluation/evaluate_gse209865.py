#!/usr/bin/env python3
"""Binary vs Sturgeon vs crossNN on GSE209865 -- this project's own registry
(registry/datasets.tsv) names this the legacy Sturgeon/nanoDx benchmark:
415 real individual nanopore low-pass WGS tumors (median ~8.6k probe calls
per sample, range 46-326660 -- real, highly variable low coverage, not
synthetic and not pooled). Never scored by any model in this project before
(confirmed gap).

Only Binary is scored, not Beta. GSE209865's raw GEO files are *already*
binary methylation calls per probe (chrom, chromStart, chromEnd,
methylation_call, probe_id) -- exactly Sturgeon's own input shape -- with no
continuous beta or (methylated_count, unmethylated_count) evidence anywhere
in what GEO published. That is precisely Binary's own feature space
(binary_transform's +1/-1/0-if-unobserved), used here with GEO's own real
calls directly -- no synthetic Poisson/threshold step at all, the most
faithful Binary evaluation on real low-coverage ONT data this project has run.
Beta need a continuous or count representation this cohort's raw data
does not provide; scoring them here would require inventing coverage this
dataset does not have, so they are skipped rather than faked.

Ground truth is GEO's own ``nanopore methylation_call`` SOFT characteristic
(the paper's own Sturgeon-derived diagnosis call, already in this project's
exact 91-class vocabulary -- checked, 48 distinct classes, 0 unmapped).
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

from neuromethyl_ont.data.array_reference import load_feature_index  # noqa: E402
from neuromethyl_ont.evaluation.classification import (  # noqa: E402
    expected_calibration_error,
    negative_log_likelihood,
    softmax_probs,
    top_k_accuracy,
)
from neuromethyl_ont.evaluation.cohort_scoring import load_torch_checkpoint  # noqa: E402
from neuromethyl_ont.evaluation.family_taxonomy import (  # noqa: E402
    UNRESOLVED,
    aggregate_to_family,
    family_metrics,
    sturgeon_model_class,
)
from neuromethyl_ont.evaluation.metrics import classification_metrics  # noqa: E402
from neuromethyl_ont.training.trainer import predict_logits  # noqa: E402


def parse_soft_labels(soft_path: Path) -> dict[str, tuple[str, str]]:
    """GSM -> (sample_title, nanopore methylation_call).

    SOFT records order ``!Sample_title`` *before* ``!Sample_geo_accession``
    within each sample block -- so a new record starts at ``Sample_title``,
    not at ``Sample_geo_accession`` (resetting ``gsm`` there, as an earlier
    version of this function did with ``title``, silently wiped every
    title before it could be used and produced zero labels)."""
    labels: dict[str, tuple[str, str]] = {}
    gsm = title = None
    with gzip.open(soft_path, "rt") as fh:
        for line in fh:
            m = re.match(r"^!Sample_title = (.+)$", line)
            if m:
                title, gsm = m.group(1).strip(), None
                continue
            m = re.match(r"^!Sample_geo_accession = (\S+)", line)
            if m:
                gsm = m.group(1)
                continue
            m = re.match(r"^!Sample_characteristics_ch1 = nanopore methylation_call: (.+)$", line)
            if m and gsm and title:
                labels[gsm] = (title, m.group(1).strip())
    return labels


def external_probs(
    path: Path, family_by_class: dict[str, str], column_to_model_class=lambda c: c
) -> tuple[dict[str, np.ndarray], list[str]]:
    table = pd.read_csv(path, sep="\t")
    score_columns = [c for c in table.columns if c not in {"sample_id", "number_probes"}]
    model_classes = [column_to_model_class(c) for c in score_columns]
    probs, families = aggregate_to_family(
        table[score_columns].to_numpy(dtype=np.float64), model_classes, family_by_class
    )
    return {s: probs[i] for i, s in enumerate(table.sample_id)}, families


def reindex(probs: np.ndarray, source: list[str], target: list[str]) -> np.ndarray:
    out = np.zeros((len(probs), len(target)), dtype=np.float64)
    for col, family in enumerate(source):
        out[:, target.index(family)] = probs[:, col]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--raw-dir", type=Path, required=True, help="datasets/GSE209865/raw/extracted")
    ap.add_argument("--soft-file", type=Path, required=True)
    ap.add_argument("--reference-dir", type=Path, required=True)
    ap.add_argument("--binary-checkpoint", "--b0-checkpoint", type=Path, required=True)
    ap.add_argument("--taxonomy", type=Path, required=True)
    ap.add_argument("--sturgeon-predictions", type=Path, default=None)
    ap.add_argument("--crossnn-predictions", type=Path, default=None)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    feature_index = load_feature_index(args.reference_dir / "feature_index.json")
    probe_pos = {probe: i for i, probe in enumerate(feature_index.probe_ids)}

    taxonomy = pd.read_csv(args.taxonomy, sep="\t")
    family_by_class = {
        str(row.model_class): str(row.family_code).strip()
        for row in taxonomy.itertuples(index=False)
        if pd.notna(row.family_code) and str(row.family_code).strip()
    }

    b0_model, b0_classes = load_torch_checkpoint(args.binary_checkpoint)
    class_to_idx = {c: i for i, c in enumerate(b0_classes)}

    soft_labels = parse_soft_labels(args.soft_file)
    unmapped = sorted({call for _, call in soft_labels.values()} - set(b0_classes))
    if unmapped:
        raise RuntimeError(f"GSE209865 class(es) not in the 91-class checkpoint: {unmapped}")

    files = sorted(args.raw_dir.glob("*.bed.gz"))
    if not files:
        ap.error(f"no *.bed.gz under {args.raw_dir}")

    n = feature_index.n_features
    x_rows, sample_ids, y_true, n_calls = [], [], [], []
    n_skipped = 0
    for path in files:
        gsm = path.name.split("_")[0]
        entry = soft_labels.get(gsm)
        if entry is None:
            n_skipped += 1
            continue
        _title, call_class = entry
        with gzip.open(path, "rt") as fh:
            bed = pd.read_csv(fh, sep=r"\s+")
        pos = bed["probe_id"].map(probe_pos)
        valid = pos.notna()
        idx = pos.loc[valid].astype(np.int64).to_numpy()
        calls = bed.loc[valid, "methylation_call"].to_numpy(dtype=np.float64)
        x_row = np.zeros(n, dtype=np.float64)
        x_row[idx] = np.where(calls > 0, 1.0, -1.0)
        x_rows.append(x_row)
        # Must match build_sturgeon_bed_gse209865.py's own sample_id exactly
        # (it uses the raw filename stem verbatim) -- reconstructing from
        # gsm+title separately risks drifting from that if a title ever has
        # characters that don't round-trip through the "<gsm>_<title>" join.
        sample_ids.append(path.name.removesuffix(".bed.gz"))
        y_true.append(class_to_idx[call_class])
        n_calls.append(int(valid.sum()))
    if n_skipped:
        print(f"WARNING: {n_skipped} raw file(s) had no SOFT label, skipped", file=sys.stderr)

    x = np.stack(x_rows)
    y_true = np.array(y_true, dtype=np.int64)
    print(f"Scoring {len(sample_ids)} labelled samples "
          f"(calls/sample within feature index: min={min(n_calls)} "
          f"median={int(np.median(n_calls))} max={max(n_calls)})", flush=True)

    b0_probs = softmax_probs(predict_logits(b0_model, x))
    true_families = np.array(
        [family_by_class.get(b0_classes[i], UNRESOLVED) for i in y_true], dtype=object
    )
    b0_fam_probs, families = aggregate_to_family(b0_probs, b0_classes, family_by_class)

    class_rows = [{
        "model": "Binary", "n_samples": len(sample_ids),
        **dict(classification_metrics(y_true, b0_probs.argmax(axis=1))),
        "top3_accuracy": top_k_accuracy(b0_probs, y_true, k=3),
        "nll": negative_log_likelihood(b0_probs, y_true),
        "ece": expected_calibration_error(b0_probs, y_true),
    }]
    family_rows = [{"model": "Binary", **family_metrics(true_families, b0_fam_probs, families)}]

    frame = pd.DataFrame({
        "sample_id": sample_ids,
        "n_calls": n_calls,
        "true_class": [b0_classes[i] for i in y_true],
        "true_family": true_families,
        "B0_predicted_class": [b0_classes[i] for i in b0_probs.argmax(axis=1)],
        "B0_true_class_probability": b0_probs[np.arange(len(y_true)), y_true],
        "B0_predicted_family": [families[i] for i in b0_fam_probs.argmax(axis=1)],
    })

    external = [
        ("Sturgeon", args.sturgeon_predictions, sturgeon_model_class),
        ("crossNN", args.crossnn_predictions, lambda c: c),
    ]
    for name, path, column_to_model_class in external:
        if path is None:
            continue
        by_sample, ext_families = external_probs(path, family_by_class, column_to_model_class)
        missing = set(sample_ids) - set(by_sample)
        if missing:
            raise RuntimeError(f"{name}: {len(missing)} GSE209865 sample(s) lack a prediction "
                                f"in {path} (e.g. {sorted(missing)[:3]})")
        stacked = reindex(np.stack([by_sample[s] for s in sample_ids]), ext_families, families)
        family_rows.append({"model": name, **family_metrics(true_families, stacked, families)})
        frame[f"{name}_predicted_family"] = [families[i] for i in stacked.argmax(axis=1)]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "gse209865_predictions.tsv", sep="\t", index=False)
    pd.DataFrame(class_rows).to_csv(
        args.output_dir / "gse209865_class_metrics.tsv", sep="\t", index=False
    )
    pd.DataFrame(family_rows).to_csv(
        args.output_dir / "gse209865_family_metrics.tsv", sep="\t", index=False
    )

    print("\n=== GSE209865, Binary 91-class accuracy ===")
    print(pd.DataFrame(class_rows)[["model", "accuracy", "top3_accuracy", "balanced_accuracy",
                                     "macro_f1", "nll", "ece"]].to_string(index=False))
    print("\n=== GSE209865, family-level accuracy ===")
    print(pd.DataFrame(family_rows)[["model", "family_accuracy", "balanced_accuracy",
                                      "macro_f1", "nll"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
