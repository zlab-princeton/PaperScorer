#!/usr/bin/env python3
"""
Generate dataset distribution figures for the LaTeX paper.

Outputs:
  tmp_latex_dir/figures/dataset1.pdf/.png  — Annual accept/reject distribution
  tmp_latex_dir/figures/dataset2.pdf/.png  — 2x2 structural statistics by year

Usage:
    python scripts/tmp_latex_dir/generate_data_dist.py
"""

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "DejaVu Sans"],
})

ACCEPT_COLOR = "#4CAF50"
REJECT_COLOR = "#F44336"
LABELSIZE = 13
TITLESIZE = 14
TICKSIZE = 11
LEGENDSIZE = 11

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "tmp_latex_dir" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Years used in the dataset
YEARS = [2020, 2021, 2022, 2023, 2025, 2026]


def load_paperstats():
    """Load paperstats and cross-reference with training data for labels."""
    stats_path = ROOT / "data" / "all_paperstats_v7.json"
    with open(stats_path) as f:
        stats = json.load(f)

    # Get labels from the baseline training+test data
    labels = {}
    for split in ["train", "test", "validation"]:
        data_path = ROOT / "data" / f"iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_{split}" / "data.json"
        if data_path.exists():
            with open(data_path) as f:
                for entry in json.load(f):
                    meta = entry.get("_metadata", {})
                    sid = meta.get("submission_id", "")
                    labels[sid] = {
                        "answer": meta.get("answer", ""),
                        "year": meta.get("year", 0),
                        "split": split,
                    }

    rows = []
    for sid, s in stats.items():
        year = int(s.get("year", 0))
        if year not in YEARS:
            continue
        if sid in labels:
            label = labels[sid]["answer"].lower()
        else:
            continue  # skip papers not in our dataset

        rows.append({
            "submission_id": sid,
            "year": year,
            "label": 1 if label == "accept" else 0,
            "num_text_tokens": s.get("num_text_tokens", 0),
            "num_pages": s.get("num_pages", 0),
            "num_figure_images": s.get("num_figure_images", 0),
            "number_of_cited_references": s.get("number_of_cited_references", 0),
        })

    return pd.DataFrame(rows)


def plot_combined(df, output_path):
    """Combined figure: (a) stacked bar + (b) 2x2 structural line plots."""
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(16, 4.5))
    gs = gridspec.GridSpec(2, 3, width_ratios=[1.5, 1, 1], hspace=0.45, wspace=0.4)

    # --- Panel (a): stacked bar, spanning left column both rows ---
    ax_bar = fig.add_subplot(gs[:, 0])

    year_accept = df[df["label"] == 1].groupby("year").size()
    year_reject = df[df["label"] == 0].groupby("year").size()
    years = YEARS
    accepts = [year_accept.get(y, 0) for y in years]
    rejects = [year_reject.get(y, 0) for y in years]
    x = np.arange(len(years))
    width = 0.55

    ax_bar.bar(x, accepts, width, label="Accept", color=ACCEPT_COLOR, edgecolor="white", linewidth=0.5)
    ax_bar.bar(x, rejects, width, bottom=accepts, label="Reject", color=REJECT_COLOR, edgecolor="white", linewidth=0.5)

    for i, (a, r) in enumerate(zip(accepts, rejects)):
        # Single count on top of each bar (accept ≈ reject since balanced)
        ax_bar.text(i, a + r + 100, str(a + r), ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color="black")

    ax_bar.set_xlabel("ICLR Year", fontsize=LABELSIZE)
    ax_bar.set_ylabel("Number of Papers", fontsize=LABELSIZE)
    # Title added via fig.text below for alignment
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([str(y) for y in years], fontsize=TICKSIZE)
    ax_bar.tick_params(axis="y", labelsize=TICKSIZE)
    ax_bar.legend(fontsize=LEGENDSIZE, loc="upper left")
    ax_bar.grid(True, linestyle="--", alpha=0.3, axis="y")

    # --- Panel (b): 2x2 structural stats in the right two columns ---
    metrics = [
        ("num_text_tokens", "Text Tokens"),
        ("num_pages", "Pages"),
        ("num_figure_images", "Num. Figures"),
        ("number_of_cited_references", "Cited References"),
    ]

    axes_b = [
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[1, 2]),
    ]

    for ax, (col, ylabel) in zip(axes_b, metrics):
        acc_means, rej_means, acc_sems, rej_sems = [], [], [], []
        for y in years:
            acc_vals = df[(df["year"] == y) & (df["label"] == 1)][col].dropna()
            rej_vals = df[(df["year"] == y) & (df["label"] == 0)][col].dropna()
            acc_means.append(acc_vals.mean() if len(acc_vals) > 0 else 0)
            rej_means.append(rej_vals.mean() if len(rej_vals) > 0 else 0)
            acc_sems.append(acc_vals.sem() if len(acc_vals) > 1 else 0)
            rej_sems.append(rej_vals.sem() if len(rej_vals) > 1 else 0)

        ax.plot(x, acc_means, marker="o", color=ACCEPT_COLOR, linewidth=2,
                markersize=5, label="Accept", zorder=5)
        ax.fill_between(x,
                        [m - s for m, s in zip(acc_means, acc_sems)],
                        [m + s for m, s in zip(acc_means, acc_sems)],
                        color=ACCEPT_COLOR, alpha=0.15)
        ax.plot(x, rej_means, marker="s", color=REJECT_COLOR, linewidth=2,
                markersize=5, label="Reject", zorder=5)
        ax.fill_between(x,
                        [m - s for m, s in zip(rej_means, rej_sems)],
                        [m + s for m, s in zip(rej_means, rej_sems)],
                        color=REJECT_COLOR, alpha=0.15)

        ax.set_ylabel(ylabel, fontsize=LABELSIZE)
        ax.set_xticks(x)
        ax.set_xticklabels([str(y) for y in years], fontsize=TICKSIZE)
        ax.tick_params(axis="y", labelsize=TICKSIZE)
        ax.grid(True, linestyle="--", alpha=0.3)

    axes_b[0].legend(fontsize=LEGENDSIZE, loc="upper left")

    # Place both titles at the same y using fig.text for alignment
    fig.canvas.draw()
    title_y = 0.97
    bar_mid = (ax_bar.get_position().x0 + ax_bar.get_position().x1) / 2
    b_mid = (axes_b[0].get_position().x0 + axes_b[1].get_position().x1) / 2
    fig.text(bar_mid, title_y, "(a) Annual Distribution",
             ha="center", fontsize=TITLESIZE, fontweight="bold")
    fig.text(b_mid, title_y, "(b) Structural Statistics by Year",
             ha="center", fontsize=TITLESIZE, fontweight="bold")

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    print("Loading data...")
    df = load_paperstats()
    print(f"Loaded {len(df)} papers across years {sorted(df['year'].unique())}")

    for y in YEARS:
        n = len(df[df["year"] == y])
        na = len(df[(df["year"] == y) & (df["label"] == 1)])
        print(f"  {y}: {n} papers ({na} accept, {n - na} reject)")

    print("\nGenerating figures...")
    plot_combined(df, OUTPUT_DIR / "dataset_overview.pdf")
    print("Done!")


if __name__ == "__main__":
    main()
