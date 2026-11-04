#!/usr/bin/env python3
"""Binary calibration (reliability diagram) and top-1-vs-top-3 accuracy, from any
cohort's evaluate_*.py output that follows the {prefix}_calibration.tsv /
{prefix}_top3.tsv / class_metrics.tsv shape (evaluate_gse109379.py,
evaluate_rapidcns2_calibration.py).

Two figures:
1. Reliability diagram -- confidence vs. observed accuracy per bin, against
   the y=x diagonal a perfectly calibrated model would sit on.
2. Top-1 vs top-3 accuracy bar, to show how many additional samples top-3
   recovers over top-1 alone.

Usage:
    python scripts/figures/plot_b0_calibration_top3.py \
        --results-dir outputs/benchmarks/rapidcns2 \
        --calibration-file rapidcns2_b0_calibration.tsv \
        --top3-file rapidcns2_b0_top3.tsv \
        --class-metrics-file rapidcns2_class_metrics.tsv \
        --dataset-label "Rapid_CNS2 (real coverage, ~2x median)" \
        --output-prefix rapidcns2 [--output-dir outputs/figures]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

from neuromethyl_ont.evaluation.plot_style import COLOR, HAIRLINE, INK, MUTED  # noqa: E402
from neuromethyl_ont.model_names import read_results_tsv  # noqa: E402


def plot_reliability_diagram(calibration: pd.DataFrame, dataset_label: str, output: Path) -> None:
    valid = calibration.dropna(subset=["confidence", "accuracy"])
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.plot([0, 1], [0, 1], color=HAIRLINE, linewidth=1.2, linestyle="--", zorder=1)
    sizes = 40 + 260 * (valid["count"] / valid["count"].max())
    ax.scatter(
        valid["confidence"], valid["accuracy"], s=sizes, color=COLOR["Binary"], zorder=3,
        edgecolor="white", linewidth=0.8,
    )
    for _, row in valid.iterrows():
        ax.annotate(
            f"n={int(row['count'])}", xy=(row["confidence"], row["accuracy"]),
            xytext=(0, 8), textcoords="offset points", ha="center", fontsize=7.5, color=MUTED,
        )
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("mean predicted confidence (bin)")
    ax.set_ylabel("observed accuracy (bin)")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(True, color=HAIRLINE, linewidth=0.8, zorder=0)
    fig.suptitle(
        f"Binary calibration — {dataset_label}", fontsize=13, fontweight="bold", x=0.02, ha="left",
        color=INK,
    )
    ax.set_title(
        "Reliability diagram, 10 confidence bins — dashed line is perfect calibration",
        fontsize=9.5, color=MUTED, loc="left", pad=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output}")


def plot_top1_vs_top3(class_metrics: pd.DataFrame, dataset_label: str, output: Path) -> None:
    row = class_metrics.loc[class_metrics["model"] == "Binary"].iloc[0]
    values = [row["accuracy"], row["top3_accuracy"]]
    labels = ["top-1", "top-3"]

    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    bars = ax.bar(labels, values, width=0.55, color=[COLOR["Binary"], COLOR["Beta"]], zorder=3)
    for bar, v in zip(bars, values, strict=True):
        ax.annotate(
            f"{v * 100:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, v),
            xytext=(0, 4), textcoords="offset points", ha="center", fontsize=10, color=INK,
        )
    ax.set_ylim(0, 1.08)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.yaxis.grid(True, color=HAIRLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0)
    fig.suptitle(
        f"Binary top-1 vs top-3 — {dataset_label}", fontsize=13, fontweight="bold", x=0.02, ha="left",
        color=INK,
    )
    ax.set_title(
        f"n={int(row['n_samples'])}, full 91-class taxonomy",
        fontsize=9.5, color=MUTED, loc="left", pad=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--calibration-file", required=True)
    ap.add_argument("--top3-file", required=True)
    ap.add_argument("--class-metrics-file", required=True)
    ap.add_argument("--dataset-label", required=True, help="e.g. 'Rapid_CNS2 (real coverage)'")
    ap.add_argument("--output-prefix", required=True, help="e.g. 'rapidcns2' -> rapidcns2_*.png")
    ap.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "figures")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    calibration = read_results_tsv(args.results_dir / args.calibration_file)
    class_metrics = read_results_tsv(args.results_dir / args.class_metrics_file)
    top3 = read_results_tsv(args.results_dir / args.top3_file)

    plot_reliability_diagram(
        calibration, args.dataset_label,
        args.output_dir / f"b0_{args.output_prefix}_reliability.png",
    )
    plot_top1_vs_top3(
        class_metrics, args.dataset_label,
        args.output_dir / f"b0_{args.output_prefix}_top1_vs_top3.png",
    )

    recovered = top3[~top3["hit_top1"] & top3["hit_topk"]]
    print(
        f"\ntop-3 recovers {len(recovered)}/{len(top3)} samples that top-1 missed "
        f"(e.g. {recovered['sample_id'].head(5).tolist()})"
    )
    recovered_path = (
        args.output_dir.parent / "tables" / f"b0_{args.output_prefix}_top3_recovered.tsv"
    )
    recovered_path.parent.mkdir(parents=True, exist_ok=True)
    recovered.to_csv(recovered_path, sep="\t", index=False)
    print(f"wrote {recovered_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
