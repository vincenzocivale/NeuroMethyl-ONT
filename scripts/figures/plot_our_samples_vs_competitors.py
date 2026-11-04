#!/usr/bin/env python3
"""Binary (ours) vs. crossNN vs. MethyLYZR, per sample, on our own low-coverage
(~10min ONT) MB/ETMR patients -- reads
`outputs/benchmarks/our_samples/mb_low_coverage_predictions_with_methylyzr.tsv`
(`merge_methylyzr_our_samples.py` output).

Sturgeon is not included -- never run on these samples in this project
(needs a separate python3.9 interpreter, not queued up). A sample not yet
scored by MethyLYZR (still extracting via bam2feather.py) renders as a
blank "pending" cell rather than 0%, so it's visually distinct from an
actual miss.

Value shown is confidence in the TRUE family (not raw top-1 confidence),
consistent with every other per-class heatmap in this project -- on this
sample set every model's top-1 call is already correct wherever it has run
(family accuracy is saturated here), so the confidence gap is the only
place these models actually differ.

Usage:
    python scripts/figures/plot_our_samples_vs_competitors.py \
        --predictions-tsv PATH/mb_low_coverage_predictions_with_methylyzr.tsv \
        --output-png outputs/figures/our_samples_heatmap.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

REPO_ROOT = Path(__file__).resolve().parents[2]

from plot_b0_per_class_heatmaps import DIVERGING_DELTA  # noqa: E402

from neuromethyl_ont.evaluation.plot_style import INK, MUTED  # noqa: E402
from neuromethyl_ont.model_names import read_results_tsv  # noqa: E402

MODELS = ["Binary", "MethyLYZR", "crossNN"]
DELTA_COL_LABEL = "Binary advantage\n(vs. best competitor)"
PREDICTIONS_TSV = (
    REPO_ROOT / "outputs/benchmarks/our_samples/mb_low_coverage_predictions_with_methylyzr.tsv"
)
OUTPUT_PNG = REPO_ROOT / "outputs/figures/our_samples_b0_methylyzr_crossnn_heatmap.png"

SEQUENTIAL_BLUE = LinearSegmentedColormap.from_list(
    "neuromethyl_blue",
    [
        "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
        "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
    ],
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions-tsv", type=Path, default=PREDICTIONS_TSV)
    ap.add_argument("--output-png", type=Path, default=OUTPUT_PNG)
    args = ap.parse_args()
    predictions_tsv, output_png = args.predictions_tsv, args.output_png
    df = read_results_tsv(predictions_tsv)
    df = df[df["model"].isin(MODELS)]
    samples = sorted(df["sample_id"].unique())
    true_family = df.drop_duplicates("sample_id").set_index("sample_id")["true_family"]

    pct = df.pivot(
        index="sample_id", columns="model", values="true_family_probability"
    ).reindex(index=samples, columns=MODELS) * 100
    correct = df.pivot(
        index="sample_id", columns="model", values="top1_correct"
    ).reindex(index=samples, columns=MODELS)

    n_rows = len(samples)
    n_cols = len(MODELS) + 1  # +1 for the "Binary advantage" delta column
    row_h, left_in, right_in, top_in, bottom_in, col_w = 0.6, 2.1, 0.9, 1.9, 0.3, 2.6
    plot_h = max(2.0, n_rows * row_h)
    fig_width = max(left_in + n_cols * col_w + right_in, 9.5)
    figsize = (fig_width, top_in + plot_h + bottom_in)
    fig, ax = plt.subplots(figsize=figsize)

    matrix = pct.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(matrix)
    im = ax.imshow(
        masked / 100, cmap=SEQUENTIAL_BLUE, vmin=0, vmax=1, aspect="auto",
        extent=(-0.5, len(MODELS) - 0.5, n_rows - 0.5, -0.5),
    )
    im.cmap.set_bad("#f0efec")

    # Binary's confidence in the true family minus the best-scoring competitor's
    # (nanmax so a still-pending MethyLYZR cell doesn't zero it out) -- one
    # glance instead of comparing three columns in your head.
    b0_col_idx = MODELS.index("Binary")
    competitor_idx = [j for j in range(len(MODELS)) if j != b0_col_idx]
    with np.errstate(invalid="ignore"):
        best_competitor = np.nanmax(matrix[:, competitor_idx], axis=1)
    delta = matrix[:, b0_col_idx] - best_competitor
    delta_has_value = ~np.isnan(delta)

    for i in range(n_rows):
        for j, model in enumerate(MODELS):
            value = matrix[i, j]
            if np.isnan(value):
                ax.text(j, i, "pending", ha="center", va="center", fontsize=8.5, color=MUTED)
                continue
            text_color = "white" if value / 100 >= 0.55 else INK
            ax.text(j, i, f"{value:.0f}%", ha="center", va="center", fontsize=10, color=text_color)
            if not bool(correct.iloc[i][model]):
                ax.add_patch(
                    Rectangle(
                        (j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="#d03b3b",
                        linewidth=2.2, zorder=5,
                    )
                )

    b0_col = MODELS.index("Binary")
    ax.add_patch(
        Rectangle(
            (b0_col - 0.5, -0.5), 1, n_rows, fill=False, edgecolor=INK,
            linewidth=1.8, zorder=4,
        )
    )

    delta_col = len(MODELS)
    if delta_has_value.any():
        delta_vmax = max(10.0, float(np.abs(delta[delta_has_value]).max()))
    else:
        delta_vmax = 10.0
    for i in range(n_rows):
        if not delta_has_value[i]:
            ax.text(delta_col, i, "pending", ha="center", va="center", fontsize=8.5, color=MUTED)
            continue
        delta_norm = np.clip(delta[i] / delta_vmax, -1, 1)
        facecolor = DIVERGING_DELTA((delta_norm + 1) / 2)
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
            fontsize=10, color=INK, fontweight="bold",
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
        [f"{s} ({true_family[s]})" for s in samples], fontsize=9.5, color=MUTED,
    )
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)

    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cbar.set_label("confidence in true family", fontsize=8.5, color=MUTED)
    cbar.ax.tick_params(length=0, labelsize=7.5, colors=MUTED)
    cbar.outline.set_visible(False)

    fig_w, fig_h = figsize
    n_scored = int(pct["MethyLYZR"].notna().sum())
    fig.suptitle(
        "Binary vs. MethyLYZR vs. crossNN — our own ETMR/MB patients",
        fontsize=13, fontweight="bold", x=0.02, y=0.99, ha="left", color=INK,
    )
    methylyzr_status = (
        "MethyLYZR scored on all patients"
        if n_scored == n_rows
        else f"MethyLYZR scored on {n_scored}/{n_rows} so far "
        "(bam2feather.py still running on the rest)"
    )
    fig.text(
        0.02, 1 - 0.5 / fig_h,
        f"{n_rows} patients, real low coverage (~10min ONT) · {methylyzr_status} · "
        "boxed column = Binary (ours) · rightmost column = Binary's edge over the best "
        "competitor · red outline = wrong top-1 call",
        fontsize=8.7, color=MUTED, ha="left",
    )
    fig.subplots_adjust(
        top=1 - top_in / fig_h, left=left_in / fig_w, right=1 - right_in / fig_w,
        bottom=bottom_in / fig_h,
    )
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=200)
    plt.close(fig)
    print(f"wrote {output_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
