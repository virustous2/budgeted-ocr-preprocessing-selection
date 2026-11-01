#!/usr/bin/env python3
"""Plot policy-agnostic controls for FixedTopK-to-oracle headroom.

Panel (a) shows the coupled statistic used in the original diagnostic:
    H = FixedTopK - Oracle
    A_P = FixedTopK - Policy

Because H and A_P share FixedTopK and A_P <= H, a positive correlation can
arise for any policy. Panel (b) therefore uses policy-to-policy gains that do
not contain FixedTopK. Each point is one dataset/setting/recognizer/budget
configuration. Inputs are cached page-level results; no OCR is rerun.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "outputs/headroom_policy_controls_cohort300_v2"
OUTPUT_STEM = PROJECT_ROOT / "IJDAR_overleaf/figures/headroom_policy_controls"

POLICY_ORDER = ["Surrogate", "Random", "NN-K"]
COMPARISON_ORDER = [
    "Surrogate_vs_Random",
    "NN-K_vs_Random",
    "Surrogate_vs_NN-K",
]
POLICY_LABELS = {
    "Surrogate": "Surrogate",
    "Random": "Random",
    "NN-K": "NN-K",
}
COMPARISON_LABELS = {
    "Surrogate_vs_Random": "Surr. vs\nRandom",
    "NN-K_vs_Random": "NN-K vs\nRandom",
    "Surrogate_vs_NN-K": "Surr. vs\nNN-K",
}
COLORS = ["#2f6f9f", "#d17b2f", "#4f8a5b"]


def draw_distribution(ax, data, category_col, categories, value_col, labels, title):
    rng = np.random.default_rng(13757)
    positions = np.arange(1, len(categories) + 1)
    values = [
        data.loc[data[category_col].eq(category), value_col].to_numpy(dtype=float)
        for category in categories
    ]

    boxes = ax.boxplot(
        values,
        positions=positions,
        widths=0.52,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.2},
        whiskerprops={"color": "0.35", "linewidth": 0.9},
        capprops={"color": "0.35", "linewidth": 0.9},
        boxprops={"color": "0.35", "linewidth": 0.9},
    )
    for box, color in zip(boxes["boxes"], COLORS):
        box.set_facecolor(color)
        box.set_alpha(0.22)

    for x, vals, color in zip(positions, values, COLORS):
        jitter = rng.uniform(-0.14, 0.14, size=len(vals))
        ax.scatter(
            x + jitter,
            vals,
            s=16,
            color=color,
            alpha=0.72,
            edgecolors="white",
            linewidths=0.25,
            zorder=3,
        )

    ax.axhline(0.0, color="0.35", linewidth=0.9, linestyle="--", zorder=1)
    ax.set_xticks(positions, [labels[c] for c in categories])
    ax.set_title(title, fontsize=10, pad=7)
    ax.set_ylabel("Spearman correlation", fontsize=9)
    ax.tick_params(axis="both", labelsize=8)
    ax.grid(axis="y", linewidth=0.35, alpha=0.45)
    ax.set_ylim(-0.62, 0.92)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--output-stem", type=Path, default=OUTPUT_STEM,
                        help="Output path without extension; writes PDF and PNG")
    args = parser.parse_args()
    summary = pd.read_csv(args.input_dir / "headroom_policy_summary_allK.csv")
    controls = pd.read_csv(args.input_dir / "headroom_nonshared_controls_allK.csv")

    coupled = summary[["policy", "spearman_headroom_vs_advantage"]].copy()
    nonshared = controls[["comparison", "spearman_headroom_vs_gain"]].copy()

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.25), constrained_layout=True)
    draw_distribution(
        axes[0],
        coupled,
        "policy",
        POLICY_ORDER,
        "spearman_headroom_vs_advantage",
        POLICY_LABELS,
        "(a) Coupled FixedTopK-based statistic",
    )
    draw_distribution(
        axes[1],
        nonshared,
        "comparison",
        COMPARISON_ORDER,
        "spearman_headroom_vs_gain",
        COMPARISON_LABELS,
        "(b) Non-shared policy-gain controls",
    )

    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(args.output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.output_stem.with_suffix('.pdf')}")
    print(f"Wrote {args.output_stem.with_suffix('.png')}")


if __name__ == "__main__":
    main()
