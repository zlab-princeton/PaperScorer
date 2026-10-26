#!/usr/bin/env python3
"""
Generate side-by-side calibration plots:
  (a) Per-bin reliability (existing style) — accuracy in each confidence bin
  (b) Cumulative threshold — accuracy on all papers with confidence >= threshold

Output: tmp_latex_dir/figures/calibration_extended_2025.pdf / .png

Uses wd_sweep_expdecay checkpoints (trained on 2020-2023+2025 only, no 2026).
"""

import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

mpl.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "DejaVu Sans"],
})

labelsize = 20
titlesize = 20
legendsize = 18
ticksize = 16

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "tmp_latex_dir" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VISION_PRED = (
    ROOT / "results/final_sweep_v7_datasweepv3/wd_sweep_expdecay"
    / "bz16_lr1e-6_wd0.001_vision/finetuned-ckpt-3990.jsonl"
)
TEXT_PRED = (
    ROOT / "results/final_sweep_v7_datasweepv3/wd_sweep_expdecay"
    / "bz16_lr1e-6_wd0.001_text/finetuned-ckpt-1594.jsonl"
)

DECISION_TOKEN_IDX = 5
BLUE = "#6098FF"
ORANGE = "#FECC81"
LINEWIDTH = 3.0
MARKERSIZE = 10


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


def load_predictions(path: Path):
    confidences, correct = [], []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            pred = extract_prediction(rec["predict"])
            label = extract_prediction(rec["label"])
            conf = math.exp(rec["token_logprobs"][DECISION_TOKEN_IDX])
            confidences.append(conf)
            correct.append(pred == label and pred != "unknown")
    return np.array(confidences), np.array(correct)


def panel_perbin(ax, text_conf, text_corr, vis_conf, vis_corr):
    """Panel (a): per-bin reliability + coverage (matching existing plot)."""
    bins = np.linspace(0.5, 1.0, 11)
    for conf, corr, color in [
        (text_conf, text_corr, BLUE),
        (vis_conf, vis_corr, ORANGE),
    ]:
        accs, mean_confs = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (conf >= lo) & (conf < hi) if hi < 1.0 else (conf >= lo) & (conf <= hi)
            if mask.sum() > 0:
                accs.append(corr[mask].mean())
                mean_confs.append(conf[mask].mean())
            else:
                accs.append(np.nan)
                mean_confs.append(np.nan)
        ax.plot(mean_confs, accs, "o-", color=color,
                linewidth=LINEWIDTH, markersize=MARKERSIZE, alpha=0.9)

    ax.plot([0.5, 1.0], [0.5, 1.0], "-", color="grey", linewidth=1.5)
    ax.set_xlabel("Confidence", fontsize=labelsize)
    ax.set_ylabel("Accuracy", fontsize=labelsize)
    ax.set_xlim(0.48, 1.02)
    ax.set_ylim(0.45, 1.05)
    ax.tick_params(axis="both", labelsize=ticksize)
    ax.grid(True, linestyle="--")

    # Coverage on right y-axis
    ax2 = ax.twinx()
    thresholds = np.linspace(0.5, 0.99, 200)
    for conf, color in [(text_conf, BLUE), (vis_conf, ORANGE)]:
        coverages = [(conf >= t).mean() for t in thresholds]
        ax2.plot(thresholds, coverages, color=color, linestyle="--",
                 linewidth=LINEWIDTH - 0.5, alpha=0.6)
    ax2.set_ylabel("Coverage", fontsize=labelsize, color="#666666")
    ax2.set_ylim(0, 1.05)
    ax2.tick_params(axis="y", labelsize=ticksize, colors="#666666")

    for t_val in [0.7, 0.8, 0.9]:
        ax.axvline(x=t_val, color="#BBBBBB", linewidth=1.0, linestyle=":",
                   alpha=0.6, zorder=0)


def panel_cumulative(ax, text_conf, text_corr, vis_conf, vis_corr):
    """Panel (b): cumulative accuracy (>= threshold) + coverage."""
    thresholds = np.linspace(0.5, 0.98, 100)

    for conf, corr, color in [
        (text_conf, text_corr, BLUE),
        (vis_conf, vis_corr, ORANGE),
    ]:
        accs = []
        for t in thresholds:
            mask = conf >= t
            if mask.sum() > 0:
                accs.append(corr[mask].mean())
            else:
                accs.append(np.nan)
        ax.plot(thresholds, accs, "-", color=color,
                linewidth=LINEWIDTH, alpha=0.9)

    # Annotate key thresholds
    for conf, corr, color, name, offset in [
        (vis_conf, vis_corr, ORANGE, "Vision", 12),
        (text_conf, text_corr, BLUE, "Text", -18),
    ]:
        for t in [0.8, 0.9]:
            mask = conf >= t
            if mask.sum() > 0:
                acc = corr[mask].mean()
                n = mask.sum()
                ax.plot(t, acc, "o", color=color, markersize=MARKERSIZE, zorder=5)
                ax.annotate(
                    f"{acc:.0%}\n(n={n})",
                    xy=(t, acc), xytext=(0, offset),
                    textcoords="offset points", ha="center", va="center",
                    fontsize=12, fontweight="bold", color=color,
                )

    ax.plot([0.5, 1.0], [0.5, 1.0], "-", color="grey", linewidth=1.5)
    ax.set_xlabel("Confidence Threshold", fontsize=labelsize)
    ax.set_ylabel("Accuracy (≥ threshold)", fontsize=labelsize)
    ax.set_xlim(0.48, 1.02)
    ax.set_ylim(0.45, 1.05)
    ax.tick_params(axis="both", labelsize=ticksize)
    ax.grid(True, linestyle="--")

    # Coverage on right y-axis
    ax2 = ax.twinx()
    for conf, color in [(text_conf, BLUE), (vis_conf, ORANGE)]:
        coverages = [(conf >= t).mean() for t in thresholds]
        ax2.plot(thresholds, coverages, color=color, linestyle="--",
                 linewidth=LINEWIDTH - 0.5, alpha=0.6)
    ax2.set_ylabel("Coverage", fontsize=labelsize, color="#666666")
    ax2.set_ylim(0, 1.05)
    ax2.tick_params(axis="y", labelsize=ticksize, colors="#666666")

    for t_val in [0.7, 0.8, 0.9]:
        ax.axvline(x=t_val, color="#BBBBBB", linewidth=1.0, linestyle=":",
                   alpha=0.6, zorder=0)


def main():
    text_conf, text_corr = load_predictions(TEXT_PRED)
    vis_conf, vis_corr = load_predictions(VISION_PRED)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(18, 7))

    # Legend
    legend_handles = [
        Line2D([0], [0], color=BLUE, marker="o", linewidth=LINEWIDTH,
               markersize=MARKERSIZE, label="Text (acc.)"),
        Line2D([0], [0], color=ORANGE, marker="o", linewidth=LINEWIDTH,
               markersize=MARKERSIZE, label="Vision (acc.)"),
        Line2D([0], [0], color="grey", linewidth=1.5, linestyle="-",
               label="y = x"),
        Line2D([0], [0], color="#888888", linewidth=LINEWIDTH - 0.5,
               linestyle="--", label="Coverage"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=4,
               fontsize=legendsize, frameon=False, bbox_to_anchor=(0.5, 1.02))

    panel_perbin(ax_a, text_conf, text_corr, vis_conf, vis_corr)
    panel_cumulative(ax_b, text_conf, text_corr, vis_conf, vis_corr)

    ax_a.set_title("(a) Per-Bin Reliability (10 bins, width=0.05)", fontsize=titlesize, fontweight="bold", pad=15)
    ax_b.set_title("(b) Cumulative Accuracy (≥ threshold)", fontsize=titlesize, fontweight="bold", pad=15)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out = OUTPUT_DIR / "calibration_extended_2025"
    plt.savefig(f"{out}.pdf", dpi=200, bbox_inches="tight")
    plt.savefig(f"{out}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}.pdf")
    print(f"Saved: {out}.png")


if __name__ == "__main__":
    main()
