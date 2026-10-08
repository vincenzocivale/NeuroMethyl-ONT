#!/usr/bin/env python3
"""Per-class error rate (not raw count) for B0 vs. Sturgeon vs. crossNN,
classes with error rate > 0 only -- one chart each for Magibu lowpass_pool
(~1x MinION, GSE209865), Magibu promethion_subclass_pool (~1x PromethION,
literature) and Rapid_CNS2 (real per-patient coverage, median ~2x, n=255).

Magibu deep_pool is deliberately EXCLUDED: the data provider flagged
that it mixes platforms, and `ValidationData/Plots/nanopore_deep_by_GPL/
accuracy_by_GPL.csv` confirms its source (GSE289246, data_type=
"nanopore_deep") is actually 68 MinION + 22 NovaSeq WGBS + 49 PromethION
samples pooled together per family -- not a clean tier. lowpass_pool's
source (GSE209865) was checked against its own series matrix and is
confirmed single-platform (GPL24106/MinION only).

Rate instead of count matters here because pool/class sizes vary a lot
(Magibu pool-rows: n=1 up to n=3 per family within a tier; Rapid_CNS2
classes: n=7 up to n=91) -- a raw miss count of 1 means something very
different for n=1 than for n=91.

Sources:
- Magibu: outputs/benchmarks/magibu_pool_comparison/magibu_pool_model_comparison_samples.tsv
  (evaluate_magibu_pools.py output) -- same one plot_magibu_coverage_comparison.py reads.
- Rapid_CNS2: outputs/benchmarks/rapidcns2_perclass.tsv (evaluate_rapidcns2.py
  output) -- already-aggregated per-class accuracy, real unpooled per-patient coverage.

CAVEAT (Magibu only, same one carried by every other Magibu-pool script):
each pool-row is 5-125 real samples averaged into one pseudo-sample beta --
a feasibility check, not single-sample accuracy. Rapid_CNS2 has no such
caveat.

Usage:
    python scripts/figures/plot_error_rate_by_class.py \
        [--output-dir outputs/figures]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

from neuromethyl_ont.evaluation.plot_style import COLOR, HAIRLINE, INK, MUTED  # noqa: E402

MODELS = ["B0", "Sturgeon", "crossNN"]


def magibu_error_rate(sub: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family, fam_rows in sub.groupby("family_code"):
        n = len(fam_rows)
        row = {"family_code": family, "n": n}
        for model in MODELS:
            row[model] = (fam_rows[f"{model}_predicted_family"] != family).mean() * 100
        rows.append(row)
    out = pd.DataFrame(rows)
    out = out.loc[out[MODELS].sum(axis=1) > 0]
    out["max_rate"] = out[MODELS].max(axis=1)
    return out.sort_values("max_rate", ascending=True).drop(columns="max_rate").reset_index(
        drop=True
    )


def rapidcns2_error_rate(perclass: pd.DataFrame, min_n: int = 5) -> pd.DataFrame:
    table = perclass[perclass["n"] >= min_n].copy()
    out = pd.DataFrame({"family_code": table["family"], "n": table["n"]})
    for model in MODELS:
        out[model] = (1 - table[model].to_numpy()) * 100
    out = out.loc[out[MODELS].sum(axis=1) > 0]
    out["max_rate"] = out[MODELS].max(axis=1)
    return out.sort_values("max_rate", ascending=True).drop(columns="max_rate").reset_index(
        drop=True
    )


def plot_error_rate(rate: pd.DataFrame, title: str, note: str, output: Path) -> None:
    if rate.empty:
        print(f"skip {output.name}: no classes with error rate > 0", file=sys.stderr)
        return

    families = rate["family_code"].tolist()
    n_families = len(families)
    n_models = len(MODELS)
    bar_h = 0.8 / n_models
    y = np.arange(n_families)

    row_h = 0.55
    left_in, right_in, top_in, bottom_in = 2.0, 0.6, 1.7, 1.1
    plot_h = max(1.2, n_families * row_h)
    figsize = (10.5, top_in + plot_h + bottom_in)
    fig, ax = plt.subplots(figsize=figsize)
    for j, model in enumerate(MODELS):
        offset = (j - (n_models - 1) / 2) * bar_h
        ax.barh(y + offset, rate[model], height=bar_h, color=COLOR[model], label=model)

    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{f} (n={n})" for f, n in zip(families, rate["n"], strict=True)],
        fontsize=9.5, color=MUTED,
    )
    ax.set_xlim(0, 108)
    ax.set_xticks(range(0, 101, 20))
    ax.set_xlabel("error rate (% of that class's pool-rows/patients misclassified)",
                   fontsize=9.5, color=MUTED)
    ax.xaxis.grid(True, color=HAIRLINE, linewidth=1)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(HAIRLINE)
    ax.tick_params(length=0)

    fig_w, fig_h = figsize
    fig.suptitle(title, fontsize=13, fontweight="bold", x=0.02, y=0.99, ha="left", color=INK)
    fig.text(0.02, 1 - 0.55 / fig_h, note, fontsize=8.7, color=MUTED, ha="left")
    fig.legend(
        loc="lower center", ncol=n_models, frameon=False, fontsize=9.5,
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.subplots_adjust(
        top=1 - top_in / fig_h, left=left_in / fig_w, right=1 - right_in / fig_w,
        bottom=bottom_in / fig_h,
    )
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(f"wrote {output}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--magibu-samples-tsv", type=Path,
        default=REPO_ROOT / "outputs/benchmarks/magibu_pool_comparison"
        / "magibu_pool_model_comparison_samples.tsv",
    )
    ap.add_argument(
        "--rapidcns2-perclass-tsv", type=Path,
        default=REPO_ROOT / "outputs" / "benchmarks" / "rapidcns2_perclass.tsv",
    )
    ap.add_argument(
        "--rapidcns2-min-n", type=int, default=5,
        help="drop Rapid_CNS2 families with fewer than this many patients (default: 5)",
    )
    ap.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "figures")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.magibu_samples_tsv, sep="\t")
    for pool, tier_label, caveat in [
        ("lowpass_pool", "Magibu lowpass_pool",
         "~1x MinION, GSE209865, verified single-platform · pooled pseudo-samples "
         "(feasibility check, not single-sample accuracy)"),
        ("promethion_subclass_pool", "Magibu promethion_subclass_pool",
         "~1x PromethION, literature · pooled pseudo-samples "
         "(feasibility check, not single-sample accuracy)"),
    ]:
        sub = df[df.benchmark == pool]
        if sub.empty:
            print(f"skip {pool}: no rows", file=sys.stderr)
            continue
        rate = magibu_error_rate(sub)
        plot_error_rate(
            rate,
            f"Error rate by class — {tier_label}",
            f"classes with 0% error across every model are omitted · {caveat}",
            args.output_dir / f"error_rate_by_class_{pool}.png",
        )

    if args.rapidcns2_perclass_tsv.exists():
        perclass = pd.read_csv(args.rapidcns2_perclass_tsv, sep="\t")
        rate = rapidcns2_error_rate(perclass, args.rapidcns2_min_n)
        plot_error_rate(
            rate,
            "Error rate by class — Rapid_CNS2 (real, median ~2x coverage, n=255)",
            f"classes with 0% error across every model are omitted · "
            f"families with n<{args.rapidcns2_min_n} excluded · real, unpooled per-patient coverage",
            args.output_dir / "error_rate_by_class_rapidcns2.png",
        )
    else:
        print(f"skip Rapid_CNS2: no file at {args.rapidcns2_perclass_tsv}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
