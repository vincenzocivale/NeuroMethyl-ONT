#!/usr/bin/env python3
"""Per-class classification-performance heatmaps: Binary (ours) vs. Sturgeon and
crossNN (competitors) -- one figure per dataset: the five Magibu pools
(deep/lowpass/nanopore_family/promethion_family/promethion_subclass, from
`outputs/benchmarks/magibu_pool_comparison/magibu_pool_model_comparison_samples.tsv`)
plus Rapid_CNS2 (real per-sample coverage, median ~2.9x, n=255, from the
already-aggregated `outputs/benchmarks/rapidcns2_perclass.tsv`).

Beta are left out on purpose (only Binary represents "ours" here) so the
comparison is a clean 3-column read: for every family present in a dataset,
what percent of that family's samples did each model call correctly?
Family codes that repeat within a Magibu pool (e.g. lowpass_pool has 3 GBM
subtypes -> GBM, 2 A_IDH subtypes -> A_IDH) give rows with n>1, so a cell
can be a real fractional accuracy, not just 0%/100%. Rows are sorted by Binary
accuracy ascending (worst first) so any gap surfaces at the top.

CAVEAT (Magibu pools only, same one evaluate_magibu_pools.py carries): each
row there is a pool of 5-125 real samples averaged into one pseudo-sample
beta -- pooling cancels noise, so those five are a feasibility check, not
single-sample accuracy. Rapid_CNS2 has no such caveat -- it's real,
unpooled per-patient coverage.

MethyLYZR is NOT included here yet: this repo only has its class priors
(`third_party/MethyLYZR/model/class_priors.csv`), not the two files its
Naive Bayes model actually needs (`betas_mean.feather`, `W_RELIEF.feather`,
downloaded separately per its README) or any read-level feather input for
our samples (it scores per-read fingerprints via `bam2feather.py` on BAMs,
not the probe/beta tables Sturgeon and crossNN take here) -- see this
session's chat for what's needed to add it.

Usage:
    python scripts/figures/plot_b0_per_class_heatmaps.py \
        [--output-dir outputs/figures]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

REPO_ROOT = Path(__file__).resolve().parents[2]

from neuromethyl_ont.evaluation.plot_style import INK, MUTED  # noqa: E402
from neuromethyl_ont.model_names import read_results_tsv  # noqa: E402

MODELS = ["Binary", "Sturgeon", "crossNN"]
MAGIBU_POOLS = [
    "deep_pool", "lowpass_pool", "nanopore_family_pool",
    "promethion_family_pool", "promethion_subclass_pool",
]

# Same brand blue as plot_style.COLOR["Binary"/"Beta"]
# (palette steps 300/450/600), extended to the full sequential ramp.
SEQUENTIAL_BLUE = LinearSegmentedColormap.from_list(
    "neuromethyl_blue",
    [
        "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
        "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
    ],
)

# Diverging red -> grey -> green for the "Binary advantage" column: red means
# the best competitor beat Binary on that class, green means Binary beat every
# competitor, so a doctor can read the sign at a glance without comparing
# the three model columns themselves.
DIVERGING_DELTA = LinearSegmentedColormap.from_list(
    "neuromethyl_delta",
    ["#b3261e", "#e2a29d", "#f0efec", "#a8d5ae", "#2f9e44"],
)
DELTA_COL_LABEL = "Binary advantage\n(vs. best competitor)"


def per_row_delta_vs_best_competitor(pct: np.ndarray) -> np.ndarray:
    """Binary's percentage minus the best-scoring competitor's, per row (in pp)."""
    b0_col = MODELS.index("Binary")
    competitor_cols = [j for j in range(len(MODELS)) if j != b0_col]
    return pct[:, b0_col] - pct[:, competitor_cols].max(axis=1)


def build_per_class_accuracy_from_samples(
    sub: pd.DataFrame,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    labels = sorted(sub["family_code"].unique())
    n_labels = len(labels)
    pct = np.zeros((n_labels, len(MODELS)))
    n_by_class = np.zeros(n_labels, dtype=int)
    for i, label in enumerate(labels):
        rows = sub[sub["family_code"] == label]
        n_by_class[i] = len(rows)
        for j, model in enumerate(MODELS):
            correct = (rows[f"{model}_predicted_family"] == label).sum()
            pct[i, j] = correct / len(rows) * 100

    order = np.lexsort((labels, pct[:, MODELS.index("Binary")]))
    return pct[order], [labels[i] for i in order], n_by_class[order]


def build_per_class_accuracy_from_perclass_table(
    perclass: pd.DataFrame, min_n: int,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    table = perclass[perclass["n"] >= min_n].copy()
    pct = table[MODELS].to_numpy(dtype=float) * 100
    labels = table["family"].tolist()
    n_by_class = table["n"].to_numpy(dtype=int)

    order = np.lexsort((labels, pct[:, MODELS.index("Binary")]))
    return pct[order], [labels[i] for i in order], n_by_class[order]


def render_class_heatmap(
    pct: np.ndarray, labels: list[str], n_by_class: np.ndarray,
    title: str, note: str, output: Path,
) -> None:
    # Rows are sorted worst-Binary-first by the caller; sort ordering itself is
    # not re-derived here.
    n_rows = len(labels)
    delta = per_row_delta_vs_best_competitor(pct)

    # Fixed-inch margins (title block, rotated top labels, row-label
    # column), so they hold steady regardless of n_rows instead of scaling
    # into either wasted whitespace (many classes) or overlap (few classes).
    row_h = 0.55
    left_in, right_in, top_in, bottom_in = 2.0, 0.9, 1.9, 0.3
    col_w = 2.6
    plot_h = max(2.0, n_rows * row_h)
    # +1 column for the "Binary advantage" delta, drawn separately below with
    # its own diverging colormap since it's not on the same 0-100% scale.
    n_cols = len(MODELS) + 1
    # The subtitle line (the longest, Rapid_CNS2's, runs ~185 chars) needs
    # ~12.5in regardless of how many columns the matrix itself would
    # otherwise need.
    fig_width = max(left_in + n_cols * col_w + right_in, 14.5)
    figsize = (fig_width, top_in + plot_h + bottom_in)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        pct / 100, cmap=SEQUENTIAL_BLUE, vmin=0, vmax=1, aspect="auto",
        extent=(-0.5, len(MODELS) - 0.5, n_rows - 0.5, -0.5),
    )

    fontsize = 10 if n_rows <= 8 else (9 if n_rows <= 12 else 8)
    for i in range(n_rows):
        for j in range(len(MODELS)):
            value = pct[i, j]
            text_color = "white" if value / 100 >= 0.55 else INK
            ax.text(
                j, i, f"{value:.0f}%", ha="center", va="center",
                fontsize=fontsize, color=text_color,
            )

    # Frame the Binary column -- it's "ours" in this comparison.
    b0_col = MODELS.index("Binary")
    ax.add_patch(
        Rectangle(
            (b0_col - 0.5, -0.5), 1, n_rows, fill=False, edgecolor=INK,
            linewidth=1.8, zorder=5,
        )
    )

    # Extra "Binary advantage" column: one glance, no mental subtraction --
    # green/"+Npp" means Binary beat the best competitor on that class, red/
    # "-Npp" means the best competitor beat Binary. Own diverging colormap
    # (symmetric around 0) since it's not a 0-100% scale like the other
    # columns.
    delta_col = len(MODELS)
    delta_vmax = max(10.0, float(np.abs(delta).max()))
    delta_norm = np.clip(delta / delta_vmax, -1, 1)
    for i in range(n_rows):
        facecolor = DIVERGING_DELTA((delta_norm[i] + 1) / 2)
        ax.add_patch(
            Rectangle(
                (delta_col - 0.5, i - 0.5), 1, 1, facecolor=facecolor,
                edgecolor="none", zorder=1,
            )
        )
        rounded = round(delta[i])
        sign = "+" if rounded > 0 else ("−" if rounded < 0 else "±")
        ax.text(
            delta_col, i, f"{sign}{abs(rounded):.0f}pp", ha="center", va="center",
            fontsize=fontsize, color=INK, fontweight="bold",
        )
    ax.add_patch(
        Rectangle(
            (delta_col - 0.5, -0.5), 1, n_rows, fill=False, edgecolor=MUTED,
            linewidth=1.0, zorder=4,
        )
    )

    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_xticks(list(range(len(MODELS))) + [delta_col])
    ax.set_xticklabels(MODELS + [DELTA_COL_LABEL], fontsize=11, color=MUTED)
    ax.xaxis.tick_top()
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(
        [f"{label} (n={total})" for label, total in zip(labels, n_by_class, strict=True)],
        fontsize=fontsize, color=MUTED,
    )
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)

    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cbar.set_label("% correct", fontsize=8.5, color=MUTED)
    cbar.ax.tick_params(length=0, labelsize=7.5, colors=MUTED)
    cbar.outline.set_visible(False)

    fig_w, fig_h = figsize
    fig.suptitle(title, fontsize=13, fontweight="bold", x=0.02, y=0.99, ha="left", color=INK)
    fig.text(0.02, 1 - 0.5 / fig_h, note, fontsize=8.7, color=MUTED, ha="left")
    fig.subplots_adjust(
        top=1 - top_in / fig_h, left=left_in / fig_w, right=1 - right_in / fig_w,
        bottom=bottom_in / fig_h,
    )
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(f"wrote {output}")


def plot_magibu_pool(pool: str, sub: pd.DataFrame, output: Path) -> None:
    pct, labels, n_by_class = build_per_class_accuracy_from_samples(sub)
    overall = {m: (sub[f"{m}_predicted_family"] == sub["family_code"]).mean() * 100 for m in MODELS}
    render_class_heatmap(
        pct, labels, n_by_class,
        title=f"Binary vs. competitors — {pool} — per-class accuracy",
        note=(
            f"{int(n_by_class.sum())} pool-samples, {len(labels)} families · overall: "
            f"Binary {overall['Binary']:.0f}%, Sturgeon {overall['Sturgeon']:.0f}%, "
            f"crossNN {overall['crossNN']:.0f}% · boxed = Binary (ours) · rightmost col = Binary's edge over best competitor · sorted by Binary accuracy"
        ),
        output=output,
    )


def plot_rapidcns2(perclass_tsv: Path, output: Path, min_n: int = 5) -> None:
    perclass = read_results_tsv(perclass_tsv)
    pct, labels, n_by_class = build_per_class_accuracy_from_perclass_table(perclass, min_n)
    overall = {
        m: (perclass[m] * perclass["n"]).sum() / perclass["n"].sum() * 100 for m in MODELS
    }
    render_class_heatmap(
        pct, labels, n_by_class,
        title="Binary vs. competitors — Rapid_CNS2 — per-class accuracy",
        note=(
            f"{int(perclass['n'].sum())} real patients (median ~2.9x coverage, unpooled), "
            f"{len(labels)} families with n≥{min_n} · overall (all classes): "
            f"Binary {overall['Binary']:.0f}%, Sturgeon {overall['Sturgeon']:.0f}%, "
            f"crossNN {overall['crossNN']:.0f}% · boxed = Binary (ours) · rightmost col = Binary's edge over best competitor · sorted by Binary accuracy"
        ),
        output=output,
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--samples-tsv", type=Path,
        default=REPO_ROOT / "outputs/benchmarks/magibu_pool_comparison"
        / "magibu_pool_model_comparison_samples.tsv",
    )
    ap.add_argument(
        "--rapidcns2-perclass-tsv", type=Path,
        default=REPO_ROOT / "outputs/benchmarks/rapidcns2_perclass.tsv",
    )
    ap.add_argument(
        "--rapidcns2-min-n", type=int, default=5,
        help="drop Rapid_CNS2 families with fewer than this many patients (default: 5)",
    )
    ap.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "figures")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = read_results_tsv(args.samples_tsv)
    for pool in MAGIBU_POOLS:
        sub = df[df.benchmark == pool]
        if sub.empty:
            print(f"skip {pool}: no rows", file=sys.stderr)
            continue
        plot_magibu_pool(pool, sub, args.output_dir / f"b0_vs_competitors_{pool}_heatmap.png")

    if args.rapidcns2_perclass_tsv.exists():
        plot_rapidcns2(
            args.rapidcns2_perclass_tsv,
            args.output_dir / "b0_vs_competitors_rapidcns2_heatmap.png",
            args.rapidcns2_min_n,
        )
    else:
        print(f"skip Rapid_CNS2: no file at {args.rapidcns2_perclass_tsv}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
