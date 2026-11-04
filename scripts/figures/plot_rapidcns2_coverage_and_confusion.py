#!/usr/bin/env python3
"""Two analyses on Rapid_CNS2 (real, unpooled per-patient coverage, n=255) --
the strongest single-sample evidence this project has, unlike the Magibu
pools (see plot_error_rate_by_class.py / plot_overall_accuracy_by_coverage.py
for why those need caveats):

1. Accuracy vs. real coverage -- bins Rapid_CNS2's actual per-patient mean
   coverage (1.2x-19.2x, not a synthetic high/low split) into quartiles and
   plots Binary/Sturgeon/crossNN family accuracy per bin. Directly tests this
   project's core claim (EVIDENCE_LINEAR_V1.md): does accuracy hold up as
   real coverage drops.

2. Confusion matrix of misclassifications -- for each model, which true
   family gets predicted as which wrong family, restricted to true classes
   with n>=5 (same threshold as rapidcns2_perclass.tsv elsewhere). Answers
   "when a model is wrong, is it a near-miss or a dangerous mix-up" --
   clinically the more useful question than an accuracy number alone.

Recomputes correctness itself (existing prediction tsvs don't carry ground
truth) by joining `rapidcns2_low_coverage_predictions.tsv`
(evaluate_rapidcns2.py output, has mean_coverage + per-model
predicted_family) with Rapid_CNS2's own ground truth (Supplementary Table 1,
via load_rapidcns2_labels -- no model re-inference needed, pure I/O) mapped
to family level via the same taxonomy_91_to_family.tsv every other script in
this project uses.

Usage:
    NEUROMETHYL_DATA_ROOT=/path/to/CNSNanoporeData \
    python scripts/figures/plot_rapidcns2_coverage_and_confusion.py \
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
sys.path.insert(0, str(REPO_ROOT / "src"))

from neuromethyl_ont.data.paths import get_data_root  # noqa: E402
from neuromethyl_ont.data.rapidcns2_labels import load_rapidcns2_labels  # noqa: E402
from neuromethyl_ont.evaluation.plot_style import COLOR, HAIRLINE, INK, MUTED  # noqa: E402
from neuromethyl_ont.model_names import read_results_tsv  # noqa: E402

MODELS = ["Binary", "Sturgeon", "crossNN"]

SEQUENTIAL_ORANGE = plt.matplotlib.colors.LinearSegmentedColormap.from_list(
    "neuromethyl_orange", ["#fbe4d5", "#f6c39a", "#f0a05f", "#eb6834", "#c94f22", "#9c3c19"],
)
SEQUENTIAL_GREEN = plt.matplotlib.colors.LinearSegmentedColormap.from_list(
    "neuromethyl_green", ["#e1f0e3", "#b8dcbe", "#8cc596", "#5ea86d", "#3d8a4f", "#296938"],
)


def build_scored_frame(
    predictions_tsv: Path, taxonomy_tsv: Path, rapid_labels_xlsx: Path,
) -> pd.DataFrame:
    taxonomy = pd.read_csv(taxonomy_tsv, sep="\t")
    family_by_class = {
        r.model_class: r.family_code for r in taxonomy.itertuples()
        if pd.notna(r.family_code) and str(r.family_code).strip()
    }
    classes = set(taxonomy.model_class.dropna())

    res = load_rapidcns2_labels(rapid_labels_xlsx, classes)
    labels = res.labels.assign(true_family=lambda d: d.label.map(family_by_class))
    labels = labels.loc[labels.true_family.notna(), ["sample_id", "true_family"]]

    preds = read_results_tsv(predictions_tsv)
    merged = preds.merge(labels, on="sample_id", how="inner")
    for model in MODELS:
        merged[f"{model}_correct"] = merged[f"{model}_predicted_family"] == merged.true_family
    return merged


def plot_accuracy_vs_coverage(scored: pd.DataFrame, output: Path) -> None:
    quartile_edges = scored["mean_coverage"].quantile([0, 0.25, 0.5, 0.75, 1.0]).to_numpy()
    quartile_edges[0] -= 1e-9  # include the minimum in the first bin
    bin_labels = [
        f"{quartile_edges[i]:.1f}–{quartile_edges[i + 1]:.1f}x" for i in range(4)
    ]
    scored = scored.copy()
    scored["coverage_bin"] = pd.cut(
        scored["mean_coverage"], bins=quartile_edges, labels=bin_labels,
    )

    rows = []
    for b in bin_labels:
        sub = scored[scored.coverage_bin == b]
        row = {"bin": b, "n": len(sub)}
        for model in MODELS:
            row[model] = sub[f"{model}_correct"].mean() * 100
        rows.append(row)
    acc = pd.DataFrame(rows)

    n_groups = len(acc)
    n_models = len(MODELS)
    bar_w = 0.75 / n_models
    x = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(10, 5.8))
    for j, model in enumerate(MODELS):
        offset = (j - (n_models - 1) / 2) * bar_w
        bars = ax.bar(x + offset, acc[model], width=bar_w, color=COLOR[model], label=model)
        for rect, value in zip(bars, acc[model], strict=True):
            ax.text(
                rect.get_x() + rect.get_width() / 2, value + 1.5, f"{value:.0f}%",
                ha="center", va="bottom", fontsize=9, color=INK,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            f"Q{i + 1}: {b}\n(n={n})"
            for i, (b, n) in enumerate(zip(acc["bin"], acc["n"], strict=True))
        ],
        fontsize=10, color=MUTED,
    )
    ax.set_ylim(0, 112)
    ax.set_xlabel("real per-patient mean coverage, quartile bins", fontsize=9.5, color=MUTED)
    ax.set_ylabel("family accuracy (%)", fontsize=9.5, color=MUTED)
    ax.yaxis.grid(True, color=HAIRLINE, linewidth=1)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(HAIRLINE)
    ax.tick_params(length=0)
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=n_models,
        frameon=False, fontsize=10,
    )

    fig.suptitle(
        "Accuracy vs. real coverage — Rapid_CNS2 (n=255, unpooled per-patient)",
        fontsize=13, fontweight="bold", x=0.02, y=0.99, ha="left", color=INK,
    )
    fig.text(
        0.02, 0.925,
        "quartile bins of real per-patient mean coverage (not a synthetic high/low split) · "
        "Q1 = lowest coverage",
        fontsize=8.7, color=MUTED, ha="left",
    )
    fig.subplots_adjust(top=0.86, bottom=0.24, left=0.08, right=0.98)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(f"wrote {output}")


def plot_confusion_matrix(scored: pd.DataFrame, model: str, min_n: int, output: Path) -> None:
    class_n = scored["true_family"].value_counts()
    eligible = set(class_n[class_n >= min_n].index)

    subset = scored.loc[scored["true_family"].isin(eligible)]
    if subset.empty:
        print(f"skip {model}: no true classes with n>={min_n}", file=sys.stderr)
        return

    pred_col = f"{model}_predicted_family"
    true_labels = sorted(subset["true_family"].unique())
    pred_labels = sorted(set(subset[pred_col].unique()) | set(true_labels))

    matrix = np.zeros((len(true_labels), len(pred_labels)), dtype=int)
    for i, t in enumerate(true_labels):
        for j, p in enumerate(pred_labels):
            matrix[i, j] = ((subset.true_family == t) & (subset[pred_col] == p)).sum()

    n_rows, n_cols = matrix.shape
    is_correct = np.zeros_like(matrix, dtype=bool)
    for i, t in enumerate(true_labels):
        for j, p in enumerate(pred_labels):
            if t == p:
                is_correct[i, j] = True

    correct_vals = matrix[is_correct]
    wrong_vals = matrix[~is_correct]
    correct_max = correct_vals.max() if correct_vals.size and correct_vals.max() > 0 else 1
    wrong_max = wrong_vals.max() if wrong_vals.size and wrong_vals.max() > 0 else 1

    rgba = np.ones((n_rows, n_cols, 4))
    for i in range(n_rows):
        for j in range(n_cols):
            value = matrix[i, j]
            if value == 0:
                rgba[i, j] = (1, 1, 1, 1)
            elif is_correct[i, j]:
                rgba[i, j] = SEQUENTIAL_GREEN(0.15 + 0.85 * value / correct_max)
            else:
                rgba[i, j] = SEQUENTIAL_ORANGE(0.15 + 0.85 * value / wrong_max)

    row_h, col_w = 0.6, 1.0
    left_in, right_in, top_in, bottom_in = 2.0, 0.6, 1.9, 1.5
    fig_w = max(6.0, left_in + n_cols * col_w + right_in)
    fig_h = top_in + max(1.5, n_rows * row_h) + bottom_in
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    ax.imshow(rgba, aspect="auto")
    for i in range(n_rows):
        for j in range(n_cols):
            value = matrix[i, j]
            if value == 0:
                continue
            r, g, b, _ = rgba[i, j]
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            text_color = "white" if luminance < 0.55 else INK
            ax.text(j, i, str(value), ha="center", va="center", fontsize=10, color=text_color)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(pred_labels, fontsize=9.5, color=MUTED, rotation=30, ha="left")
    ax.xaxis.tick_top()
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(
        [f"{t} (n={int(class_n[t])})" for t in true_labels], fontsize=9.5, color=MUTED,
    )
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)
    ax.set_xlabel("predicted as", fontsize=9.5, color=MUTED)
    ax.xaxis.set_label_position("top")
    ax.set_ylabel("true family", fontsize=9.5, color=MUTED)

    legend_handles = [
        plt.matplotlib.patches.Patch(
            facecolor=SEQUENTIAL_GREEN(0.7), label="correct (true = predicted)",
        ),
        plt.matplotlib.patches.Patch(facecolor=SEQUENTIAL_ORANGE(0.7), label="misclassified"),
    ]
    ax.legend(
        handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.15),
        ncol=2, frameon=False, fontsize=9.5,
    )

    fig_w2, fig_h2 = fig_w, fig_h
    fig.suptitle(
        f"{model} confusion matrix — Rapid_CNS2 (true → predicted)",
        fontsize=13, fontweight="bold", x=0.02, y=0.99, ha="left", color=INK,
    )
    fig.text(
        0.02, 1 - 0.55 / fig_h2,
        f"only true classes with n≥{min_n} shown · green = correct, orange = misclassified",
        fontsize=8.7, color=MUTED, ha="left",
    )
    fig.subplots_adjust(
        top=1 - top_in / fig_h2, left=left_in / fig_w2, right=1 - right_in / fig_w2,
        bottom=bottom_in / fig_h2,
    )
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(f"wrote {output}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--predictions-tsv", type=Path,
        default=REPO_ROOT / "outputs/benchmarks/rapidcns2_model_comparison"
        / "rapidcns2_low_coverage_predictions.tsv",
    )
    ap.add_argument(
        "--taxonomy-tsv", type=Path, default=None,
        help="defaults to $NEUROMETHYL_DATA_ROOT/derived/evaluation/magibu_v1/"
        "taxonomy/taxonomy_91_to_family.tsv",
    )
    ap.add_argument(
        "--rapid-labels-xlsx", type=Path, default=None,
        help="defaults to $NEUROMETHYL_DATA_ROOT/datasets/RapidCNS2/metadata/"
        "Supplementary_Tables_1_2.xlsx",
    )
    ap.add_argument("--min-n", type=int, default=5, help="min patients per true class (default 5)")
    ap.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "figures")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data_root = get_data_root()
    taxonomy_tsv = args.taxonomy_tsv or (
        data_root / "derived/evaluation/magibu_v1/taxonomy/taxonomy_91_to_family.tsv"
    )
    rapid_labels_xlsx = args.rapid_labels_xlsx or (
        data_root / "datasets/RapidCNS2/metadata/Supplementary_Tables_1_2.xlsx"
    )

    scored = build_scored_frame(args.predictions_tsv, taxonomy_tsv, rapid_labels_xlsx)
    print(f"scored {len(scored)} Rapid_CNS2 patients with ground truth + real coverage")

    plot_accuracy_vs_coverage(scored, args.output_dir / "rapidcns2_accuracy_vs_coverage.png")
    for model in MODELS:
        plot_confusion_matrix(
            scored, model, args.min_n,
            args.output_dir / f"rapidcns2_confusion_errors_{model.lower()}.png",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
