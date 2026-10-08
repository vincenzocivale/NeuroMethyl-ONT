#!/usr/bin/env python3
"""High-coverage vs. low-coverage model comparison on the Magibu
pools, two chart types per coverage tier (deep_pool = high coverage,
lowpass_pool = ~1x low coverage) -- reads
`outputs/benchmarks/magibu_pool_comparison/magibu_pool_model_comparison_samples.tsv`
(`evaluate_magibu_pools.py` output), same source table as
`plot_b0_per_class_heatmaps.py`.

1. Per-class accuracy bar chart -- for every family present in the tier,
   what percent of that family's pool-rows did each of the 5 models
   (B0/C0/E1/Sturgeon/crossNN) call correctly.
2. Per-model misclassification frequency -- for every (model, family) pair
   with at least one wrong call, how many of that family's pool-rows the
   model missed. deep_pool has one pool-row per family (5 families, no
   repeats -- counts are 0 or 1 there); lowpass_pool has some families
   split across several pools (e.g. GBM: MES/RTK_I/RTK_II, A_IDH: HG/non-HG,
   MB: G4/WNT), so counts there can exceed 1.

CAVEAT (same one evaluate_magibu_pools.py and plot_b0_per_class_heatmaps.py
carry): each pool-row is 5-125 real samples averaged into one pseudo-sample
beta -- pooling cancels noise, so this is a feasibility check, not
single-sample accuracy at either coverage tier.

Usage:
    python scripts/figures/plot_magibu_coverage_comparison.py \
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
COVERAGE_TIERS = {
    "deep_pool": "high coverage (deep_pool)",
    "lowpass_pool": "low coverage (~1x, lowpass_pool)",
}


def per_class_accuracy(sub: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family, fam_rows in sub.groupby("family_code"):
        n = len(fam_rows)
        row = {"family_code": family, "n": n}
        for model in MODELS:
            row[model] = (fam_rows[f"{model}_predicted_family"] == family).mean() * 100
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values(MODELS[0], ascending=False).reset_index(drop=True)


def per_class_miss_counts(sub: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family, fam_rows in sub.groupby("family_code"):
        n = len(fam_rows)
        row = {"family_code": family, "n": n}
        for model in MODELS:
            row[model] = int((fam_rows[f"{model}_predicted_family"] != family).sum())
        rows.append(row)
    out = pd.DataFrame(rows)
    out["total_misses"] = out[MODELS].sum(axis=1)
    out = out.loc[out["total_misses"] > 0]
    return out.sort_values("total_misses", ascending=True).reset_index(drop=True)


def plot_accuracy_bar(acc: pd.DataFrame, pool: str, tier_label: str, output: Path) -> None:
    families = acc["family_code"].tolist()
    n_families = len(families)
    n_models = len(MODELS)
    bar_w = 0.8 / n_models
    x = np.arange(n_families)

    fig_w = max(9.5, 1.05 * n_families)
    fig, ax = plt.subplots(figsize=(fig_w, 5.2))
    for j, model in enumerate(MODELS):
        offset = (j - (n_models - 1) / 2) * bar_w
        ax.bar(x + offset, acc[model], width=bar_w, color=COLOR[model], label=model)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{f}\n(n={n})" for f, n in zip(families, acc["n"], strict=True)],
        fontsize=9, color=MUTED,
    )
    ax.set_ylim(0, 108)
    ax.set_ylabel("% correct (top-1 family call)", fontsize=9.5, color=MUTED)
    ax.yaxis.grid(True, color=HAIRLINE, linewidth=1)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(HAIRLINE)
    ax.tick_params(length=0)
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=n_models,
        frameon=False, fontsize=9.5,
    )

    fig.suptitle(
        f"Per-class accuracy — {tier_label}", fontsize=13, fontweight="bold",
        x=0.02, y=0.99, ha="left", color=INK,
    )
    fig.text(
        0.02, 0.925,
        f"{int(acc['n'].sum())} pool-rows, {n_families} families · pooled pseudo-samples "
        "(feasibility check, not single-sample accuracy) · sorted by B0 accuracy",
        fontsize=8.7, color=MUTED, ha="left",
    )
    fig.subplots_adjust(top=0.86, bottom=0.24, left=0.06, right=0.98)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(f"wrote {output}")


def plot_miss_frequency(miss: pd.DataFrame, pool: str, tier_label: str, output: Path) -> None:
    if miss.empty:
        print(f"skip {pool} miss-frequency: no misclassifications", file=sys.stderr)
        return

    families = miss["family_code"].tolist()
    n_families = len(families)
    n_models = len(MODELS)
    bar_h = 0.8 / n_models
    y = np.arange(n_families)

    # Fixed-inch margins (title block, x-axis label, legend row), so they
    # hold steady regardless of n_families.
    row_h = 0.55
    left_in, right_in, top_in, bottom_in = 2.0, 0.6, 1.7, 1.1
    plot_h = max(1.2, n_families * row_h)
    figsize = (8.5, top_in + plot_h + bottom_in)
    fig, ax = plt.subplots(figsize=figsize)
    for j, model in enumerate(MODELS):
        offset = (j - (n_models - 1) / 2) * bar_h
        ax.barh(y + offset, miss[model], height=bar_h, color=COLOR[model], label=model)

    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{f} (n={n})" for f, n in zip(families, miss["n"], strict=True)],
        fontsize=9.5, color=MUTED,
    )
    max_count = int(miss[MODELS].to_numpy().max())
    ax.set_xticks(range(0, max_count + 2))
    ax.set_xlim(0, max_count + 1)
    ax.set_xlabel("misclassified pool-rows (count)", fontsize=9.5, color=MUTED)
    ax.xaxis.grid(True, color=HAIRLINE, linewidth=1)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(HAIRLINE)
    ax.tick_params(length=0)

    fig_w, fig_h = figsize
    fig.suptitle(
        f"Misclassifications by model — {tier_label}", fontsize=13, fontweight="bold",
        x=0.02, y=0.99, ha="left", color=INK,
    )
    fig.text(
        0.02, 1 - 0.55 / fig_h,
        "families with 0 misses across every model are omitted · "
        "only classes with real repeats within a tier can show a count above 1",
        fontsize=8.7, color=MUTED, ha="left",
    )
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
        "--samples-tsv", type=Path,
        default=REPO_ROOT / "outputs/benchmarks/magibu_pool_comparison"
        / "magibu_pool_model_comparison_samples.tsv",
    )
    ap.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "figures")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.samples_tsv, sep="\t")
    for pool, tier_label in COVERAGE_TIERS.items():
        sub = df[df.benchmark == pool]
        if sub.empty:
            print(f"skip {pool}: no rows", file=sys.stderr)
            continue

        acc = per_class_accuracy(sub)
        plot_accuracy_bar(
            acc, pool, tier_label,
            args.output_dir / f"magibu_{pool}_class_accuracy_bar.png",
        )

        miss = per_class_miss_counts(sub)
        plot_miss_frequency(
            miss, pool, tier_label,
            args.output_dir / f"magibu_{pool}_misclassification_frequency.png",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
