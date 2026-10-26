#!/usr/bin/env python3
"""
Scatter plot: p(accept) vs pct_rating (human review percentile).
  x-axis: p(accept) = exp(logprob) if pred=accept, else 1 - exp(logprob)
  y-axis: pct_rating from test metadata

3×2 grid: rows = Overall / ICLR 2025 / ICLR 2026, cols = Text / Vision.

Uses optim_search_2026 best checkpoints:
  Text:   bz16_lr0.5e-6_wd0.001_text / ckpt-2644
  Vision: bz16_lr1e-6_wd0.001_vision / ckpt-5296
"""

import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

mpl.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "DejaVu Sans"],
})

labelsize = 16
titlesize = 18
legendsize = 13
ticksize = 13

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "tmp_latex_dir" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RUNS = {
    "Text": {
        "pred_path": ROOT / "results/final_sweep_v7_datasweepv3/optim_search_2026"
                          / "bz16_lr0.5e-6_wd0.001_text/finetuned-ckpt-2644.jsonl",
        "test_data": ROOT / "data/iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_test/data.json",
        "color": "#6098FF",
        "marker": "o",
    },
    "Vision": {
        "pred_path": ROOT / "results/final_sweep_v7_datasweepv3/optim_search_2026"
                          / "bz16_lr1e-6_wd0.001_vision/finetuned-ckpt-5296.jsonl",
        "test_data": ROOT / "data/iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_labelfix_v7_filtered_filtered24480_test/data.json",
        "color": "#FECC81",
        "marker": "s",
    },
}

DECISION_TOKEN_IDX = 5


def extract_prediction(text: str) -> str:
    t = text.lower().strip()
    if "\\boxed{accept}" in t or "boxed{accept}" in t:
        return "accept"
    if "\\boxed{reject}" in t or "boxed{reject}" in t:
        return "reject"
    if "accept" in t:
        return "accept"
    if "reject" in t:
        return "reject"
    return "unknown"


def load_data(pred_path: Path, test_path: Path):
    """Load predictions + test metadata, return p(accept), pct_rating, year arrays."""
    preds = []
    with open(pred_path) as f:
        for line in f:
            preds.append(json.loads(line))

    test_data = json.load(open(test_path))

    n = min(len(preds), len(test_data))
    p_accept = []
    pct_ratings = []
    years = []

    for i in range(n):
        rec = preds[i]
        meta = test_data[i].get("_metadata", {})
        pct_rating = meta.get("pct_rating")
        year = meta.get("year")
        if pct_rating is None or year is None:
            continue

        pred = extract_prediction(rec["predict"])
        logprob = rec["token_logprobs"][DECISION_TOKEN_IDX]
        conf = math.exp(logprob)

        if pred == "accept":
            pa = conf
        elif pred == "reject":
            pa = 1.0 - conf
        else:
            continue

        p_accept.append(pa)
        pct_ratings.append(pct_rating)
        years.append(year)

    return np.array(p_accept), np.array(pct_ratings), np.array(years)


def plot_panel(ax, pa, pr, color, marker, title):
    """Plot a single scatter panel with linear fit, binned means, and correlation."""
    if len(pa) < 3:
        ax.set_title(f"{title} (n=0)", fontsize=titlesize, fontweight="bold")
        return

    r, p_val = stats.pearsonr(pa, pr)
    rho, sp_val = stats.spearmanr(pa, pr)

    print(f"  {title}: n={len(pa)}, Pearson r={r:.4f}, Spearman ρ={rho:.4f}")

    # Scatter
    #ax.scatter(pa, pr, c=color, marker=marker,
    #           alpha=0.25, s=20, edgecolors="none")

    # Linear fit
    slope, intercept = np.polyfit(pa, pr, 1)
    x_fit = np.linspace(0, 1, 100)
    y_fit = slope * x_fit + intercept
    ax.plot(x_fit, y_fit, color="red", linewidth=2.5,
            linestyle="--", label=f"Linear fit (r={r:.3f})")

    # Binned means
    bin_edges = np.linspace(0, 1, 21)
    bin_centers, bin_means = [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (pa >= lo) & (pa < hi) if hi < 1.0 else (pa >= lo) & (pa <= hi)
        if mask.sum() >= 5:
            bin_centers.append(pa[mask].mean())
            bin_means.append(pr[mask].mean())
    ax.plot(bin_centers, bin_means, "D-", color="darkred",
            markersize=5, linewidth=1.8, alpha=0.8, label="Binned mean")

    ax.set_xlabel("p(accept)", fontsize=labelsize)
    ax.set_ylabel("pct_rating", fontsize=labelsize)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"{title}", fontsize=titlesize, fontweight="bold")
    ax.tick_params(axis="both", labelsize=ticksize)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=legendsize - 2, loc="upper left")

    ax.text(0.95, 0.05,
            f"r = {r:.3f}, ρ = {rho:.3f}\nn = {len(pa)}",
            transform=ax.transAxes, fontsize=12,
            verticalalignment="bottom", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="gray", alpha=0.8))


def main():
    # Load all data
    all_data = {}
    for name, cfg in RUNS.items():
        pa, pr, yrs = load_data(cfg["pred_path"], cfg["test_data"])
        all_data[name] = (pa, pr, yrs, cfg["color"], cfg["marker"])

    # 3×2 grid: rows = Overall / 2025 / 2026, cols = Text / Vision
    row_labels = [
        ("Overall", None),
        ("ICLR 2025", 2025),
        ("ICLR 2026", 2026),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(16, 20))

    for row_idx, (row_name, year_filter) in enumerate(row_labels):
        for col_idx, modality in enumerate(["Text", "Vision"]):
            ax = axes[row_idx, col_idx]
            pa, pr, yrs, color, marker = all_data[modality]

            if year_filter is not None:
                mask = yrs == year_filter
                pa_f, pr_f = pa[mask], pr[mask]
            else:
                pa_f, pr_f = pa, pr

            title = f"{modality} — {row_name}"
            plot_panel(ax, pa_f, pr_f, color, marker, title)

    plt.tight_layout()

    out = OUTPUT_DIR / "calibration_human_corr"
    plt.savefig(f"{out}.pdf", dpi=200, bbox_inches="tight")
    plt.savefig(f"{out}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out}.pdf")
    print(f"Saved: {out}.png")


if __name__ == "__main__":
    main()
