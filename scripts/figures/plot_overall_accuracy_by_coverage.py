#!/usr/bin/env python3
"""Overall (not per-class) B0 vs. Sturgeon vs. crossNN family accuracy,
one bar chart with three groups, all platform-verified:

- Magibu lowpass_pool (GSE209865, confirmed single-platform MinION/GPL24106
  via its series matrix -- ~1x per-CpG nanopore)
- Magibu promethion_subclass_pool (literature PromethION samples, also
  ~1x per-CpG per the data provider -- a different platform, not a "high
  coverage" tier)
- Rapid_CNS2 (real per-patient coverage, median ~2x, n=255)

Both from `outputs/benchmarks/magibu_pool_comparison/
magibu_pool_model_comparison_metrics.tsv` / `.../rapidcns2_model_comparison/
rapidcns2_model_comparison_metrics.tsv`.

Magibu deep_pool is deliberately EXCLUDED, not just uncertain: the
data provider flagged it as mixing platforms, and
`ValidationData/Plots/nanopore_deep_by_GPL/accuracy_by_GPL.csv` confirms
its source (GSE289246, data_type="nanopore_deep") is actually 68 MinION +
22 NovaSeq WGBS + 49 PromethION samples pooled together per family --
not a clean nanopore-only, let alone "high coverage", tier. There is no
clean pooled high-coverage counterpart in the Magibu data; Rapid_CNS2 (real,
unpooled, coverage up to ~19x per patient) is the only genuine coverage
signal here.

Magibu pools remain pools of 5-125 real samples averaged into one
pseudo-sample beta each -- a feasibility check, not single-sample accuracy
(same caveat as evaluate_magibu_pools.py / plot_magibu_coverage_comparison.py).
Rapid_CNS2 has no such caveat: real, unpooled per-patient coverage.

Usage:
    python scripts/figures/plot_overall_accuracy_by_coverage.py \
        [--output-dir outputs/figures]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

from neuromethyl_ont.evaluation.plot_style import COLOR, HAIRLINE, INK, MUTED  # noqa: E402

MODELS = ["B0", "Sturgeon", "crossNN"]
POOL_GROUPS = [
    ("lowpass_pool", "Magibu lowpass_pool\n(~1x MinION, GSE209865, verified single-platform)"),
    ("promethion_subclass_pool", "Magibu promethion_subclass_pool\n(~1x PromethION, literature)"),
]
RAPIDCNS2_GROUP = "Rapid_CNS2\n(real, median ~2x, n=255)"


def load_accuracy(
    magibu_metrics_tsv: Path, rapidcns2_metrics_tsv: Path,
) -> pd.DataFrame:
    magibu = pd.read_csv(magibu_metrics_tsv, sep="\t")
    rapid = pd.read_csv(rapidcns2_metrics_tsv, sep="\t")

    rows = []
    for pool, group in POOL_GROUPS:
        sub = magibu[magibu.pool == pool].set_index("model")
        rows.append({"group": group, **{m: sub.loc[m, "family_accuracy"] * 100 for m in MODELS}})
    rapid_row = rapid.set_index("model")
    rows.append(
        {"group": RAPIDCNS2_GROUP, **{m: rapid_row.loc[m, "family_accuracy"] * 100 for m in MODELS}}
    )
    return pd.DataFrame(rows)


def plot(acc: pd.DataFrame, output: Path) -> None:
    n_groups = len(acc)
    n_models = len(MODELS)
    bar_w = 0.75 / n_models
    x = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    for j, model in enumerate(MODELS):
        offset = (j - (n_models - 1) / 2) * bar_w
        bars = ax.bar(x + offset, acc[model], width=bar_w, color=COLOR[model], label=model)
        for rect, value in zip(bars, acc[model], strict=True):
            ax.text(
                rect.get_x() + rect.get_width() / 2, value + 1.5, f"{value:.0f}%",
                ha="center", va="bottom", fontsize=9, color=INK,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(acc["group"], fontsize=10, color=MUTED)
    ax.set_ylim(0, 112)
    ax.set_ylabel("overall family accuracy (%)", fontsize=9.5, color=MUTED)
    ax.yaxis.grid(True, color=HAIRLINE, linewidth=1)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(HAIRLINE)
    ax.tick_params(length=0)
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=n_models,
        frameon=False, fontsize=10,
    )

    fig.suptitle(
        "Overall accuracy — B0 vs. Sturgeon vs. crossNN (platform-verified pools only)",
        fontsize=13, fontweight="bold", x=0.02, y=0.99, ha="left", color=INK,
    )
    fig.text(
        0.02, 0.925,
        "Magibu deep_pool excluded (mixes MinION+WGBS+PromethION, per the data provider) · Magibu "
        "pools are pooled pseudo-samples (feasibility check, not single-sample accuracy)",
        fontsize=8.7, color=MUTED, ha="left",
    )
    fig.text(
        0.02, 0.885,
        "Rapid_CNS2: real, unpooled per-patient coverage",
        fontsize=8.7, color=MUTED, ha="left",
    )
    fig.subplots_adjust(top=0.82, bottom=0.22, left=0.07, right=0.98)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(f"wrote {output}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--magibu-metrics-tsv", type=Path,
        default=REPO_ROOT / "outputs/benchmarks/magibu_pool_comparison"
        / "magibu_pool_model_comparison_metrics.tsv",
    )
    ap.add_argument(
        "--rapidcns2-metrics-tsv", type=Path,
        default=REPO_ROOT / "outputs/benchmarks/rapidcns2_model_comparison"
        / "rapidcns2_model_comparison_metrics.tsv",
    )
    ap.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "figures")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    acc = load_accuracy(args.magibu_metrics_tsv, args.rapidcns2_metrics_tsv)
    plot(acc, args.output_dir / "overall_accuracy_verified_pools_and_rapidcns2.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
