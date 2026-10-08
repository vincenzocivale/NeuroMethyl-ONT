#!/usr/bin/env python3
"""What a single B0 top-3 prediction looks like, for two real Rapid_CNS2
samples (evaluate_rapidcns2_calibration.py's per-sample top3 table):

- a top-1 "close call" (ID_157, SCHW) where the top-2 candidate is nearly
  tied with the winner -- top-3 here is just showing the runner-up.
- a top-1 miss the top-3 list recovers (ID_110, PLEX_PED_B) -- the true
  class is buried at rank 3 with a tiny probability, illustrating both what
  "top-3 recovers this sample" means and how little probability mass B0
  gives the true class even when it's technically inside the top-3 (ties
  into the reliability-diagram overconfidence finding).

Probabilities span many orders of magnitude (this model saturates its
softmax hard, see plot_b0_calibration_top3.py), so bars are on a log y-axis
-- a linear axis would render rank 2/3 invisible for most samples.

Usage:
    python scripts/figures/plot_single_prediction_top3.py \
        --top3-file outputs/benchmarks/rapidcns2/rapidcns2_b0_top3.tsv \
        [--sample-hit ID_157] [--sample-recovered ID_110] \
        [--output-dir outputs/figures]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

from neuromethyl_ont.evaluation.plot_style import COLOR, HAIRLINE, INK, MUTED  # noqa: E402

CORRECT_COLOR = "#2f9e5c"


def plot_sample(row: pd.Series, title: str, subtitle: str, ax) -> None:
    ranks = [1, 2, 3]
    classes = [row[f"top{k}_class"] for k in ranks]
    probs = [max(row[f"top{k}_probability"], 1e-6) for k in ranks]
    colors = [
        CORRECT_COLOR if cls == row["true_class"] else COLOR["B0"] for cls in classes
    ]
    bars = ax.bar([f"top-{k}" for k in ranks], probs, color=colors, width=0.6, zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 2)
    for bar, cls, prob in zip(bars, classes, probs, strict=True):
        ax.annotate(
            f"{cls}\n{prob:.1e}" if prob < 1e-3 else f"{cls}\n{prob:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, prob),
            xytext=(0, 5), textcoords="offset points", ha="center", va="bottom",
            fontsize=8.5, color=INK,
        )
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_ylabel("predicted probability (log scale)")
    ax.yaxis.grid(True, color=HAIRLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(subtitle, fontsize=9.5, color=MUTED, loc="left", pad=8)
    ax.text(
        0.98, 0.95, f"true class: {row['true_class']}", transform=ax.transAxes,
        ha="right", va="top", fontsize=9, color=CORRECT_COLOR, fontweight="bold",
    )
    ax.set_title(title, fontsize=11, fontweight="bold", loc="left", pad=28, color=INK)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--top3-file", type=Path, required=True)
    ap.add_argument("--sample-hit", default="ID_157", help="top-1 correct, close call")
    ap.add_argument("--sample-recovered", default="ID_110", help="top-1 wrong, top-3 recovers")
    ap.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "figures")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    table = pd.read_csv(args.top3_file, sep="\t").set_index("sample_id")
    hit_row = table.loc[args.sample_hit]
    recovered_row = table.loc[args.sample_recovered]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.2))
    plot_sample(
        hit_row, f"{args.sample_hit} — top-1 correct, close call",
        "Rank-2 nearly ties the winner — top-3 shows the runner-up", ax1,
    )
    plot_sample(
        recovered_row, f"{args.sample_recovered} — top-3 recovers a top-1 miss",
        "True class ranked 3rd — buried, but still on the list", ax2,
    )
    fig.suptitle(
        "What a B0 top-3 prediction looks like (Rapid_CNS2)", fontsize=14, fontweight="bold",
        x=0.02, ha="left", color=INK, y=1.03,
    )
    fig.tight_layout()
    output = args.output_dir / "b0_single_prediction_top3.png"
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
