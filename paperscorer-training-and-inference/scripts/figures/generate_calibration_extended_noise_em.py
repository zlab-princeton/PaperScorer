#!/usr/bin/env python3
"""
Generate side-by-side calibration plots for noise-EM model:
  (a) Per-bin reliability — accuracy in each confidence bin
  (b) Cumulative threshold — accuracy on all papers with confidence >= threshold

Uses MLP classification head predictions (prob field as confidence).
Compares text noise-EM (trainagreeing) against text SFT baseline (wd_sweep_expdecay).

Output: tmp_latex_dir/figures/calibration_extended_noise_em.pdf / .png
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

# Noise-EM (MLP head, trainagreeing, 2025 data)
NOISE_EM_PATH = ROOT / "results/lr_experiment_v7/text_trainagreeing_no2024_noise_em/generated_predictions.jsonl"

# SFT text baseline for comparison (wd_sweep_expdecay, 2025 data)
SFT_TEXT_PATH = (
    ROOT / "results/final_sweep_v7_datasweepv3/wd_sweep_expdecay"
    / "bz16_lr1e-6_wd0.001_text/finetuned-ckpt-1594.jsonl"
)

DECISION_TOKEN_IDX = 5
BLUE = "#6098FF"
ORANGE = "#FECC81"
GREEN = "#66BB6A"
LINEWIDTH = 3.0
MARKERSIZE = 10


def extract_prediction(text: str) -> str:
    t = text.lower().strip()
    if "accept" in t: return "accept"
    if "reject" in t: return "reject"
    return "unknown"


def load_sft_predictions(path: Path):
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


def load_mlp_predictions(path: Path):
    confidences, correct = [], []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            pred = int(rec["pred"])
            label = int(rec["label"])
            prob = float(rec["prob"])
            # prob is P(accept) = P(class=1). Confidence = prob if pred=1, else 1-prob
            conf = prob if pred == 1 else 1.0 - prob
            correct.append(pred == label)
            confidences.append(conf)
    return np.array(confidences), np.array(correct)


def panel_perbin(ax, datasets):
    bins = np.linspace(0.5, 1.0, 11)
    for conf, corr, color, label in datasets:
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

    ax2 = ax.twinx()
    thresholds = np.linspace(0.5, 0.99, 200)
    for conf, _, color, _ in datasets:
        coverages = [(conf >= t).mean() for t in thresholds]
        ax2.plot(thresholds, coverages, color=color, linestyle="--",
                 linewidth=LINEWIDTH - 0.5, alpha=0.6)
    ax2.set_ylabel("Coverage", fontsize=labelsize, color="#666666")
    ax2.set_ylim(0, 1.05)
    ax2.tick_params(axis="y", labelsize=ticksize, colors="#666666")

    for t_val in [0.7, 0.8, 0.9]:
        ax.axvline(x=t_val, color="#BBBBBB", linewidth=1.0, linestyle=":",
                   alpha=0.6, zorder=0)


def panel_cumulative(ax, datasets):
    thresholds = np.linspace(0.5, 0.98, 100)

    for conf, corr, color, label in datasets:
        accs = []
        for t in thresholds:
            mask = conf >= t
            if mask.sum() > 0:
                accs.append(corr[mask].mean())
            else:
                accs.append(np.nan)
        ax.plot(thresholds, accs, "-", color=color,
                linewidth=LINEWIDTH, alpha=0.9)

        for t in [0.8, 0.9]:
            mask = conf >= t
            if mask.sum() > 0:
                acc = corr[mask].mean()
                n = mask.sum()
                ax.plot(t, acc, "o", color=color, markersize=MARKERSIZE, zorder=5)
                offset = 12 if color != BLUE else -18
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

    ax2 = ax.twinx()
    for conf, _, color, _ in datasets:
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
    sft_conf, sft_corr = load_sft_predictions(SFT_TEXT_PATH)
    noise_conf, noise_corr = load_mlp_predictions(NOISE_EM_PATH)

    print(f"SFT Text: {len(sft_conf)} predictions, acc={sft_corr.mean():.1%}")
    print(f"Noise-EM: {len(noise_conf)} predictions, acc={noise_corr.mean():.1%}")

    datasets = [
        (sft_conf, sft_corr, BLUE, "Text SFT"),
        (noise_conf, noise_corr, GREEN, "Text Noise-EM"),
    ]

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(18, 7))

    legend_handles = [
        Line2D([0], [0], color=BLUE, marker="o", linewidth=LINEWIDTH,
               markersize=MARKERSIZE, label="Text SFT (acc.)"),
        Line2D([0], [0], color=GREEN, marker="o", linewidth=LINEWIDTH,
               markersize=MARKERSIZE, label="Text Noise-EM (acc.)"),
        Line2D([0], [0], color="grey", linewidth=1.5, linestyle="-",
               label="y = x"),
        Line2D([0], [0], color="#888888", linewidth=LINEWIDTH - 0.5,
               linestyle="--", label="Coverage"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=4,
               fontsize=legendsize, frameon=False, bbox_to_anchor=(0.5, 1.02))

    panel_perbin(ax_a, datasets)
    panel_cumulative(ax_b, datasets)

    ax_a.set_title("(a) Per-Bin Reliability (10 bins, width=0.05)", fontsize=titlesize, fontweight="bold", pad=15)
    ax_b.set_title("(b) Cumulative Accuracy (≥ threshold)", fontsize=titlesize, fontweight="bold", pad=15)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out = OUTPUT_DIR / "calibration_extended_noise_em"
    plt.savefig(f"{out}.pdf", dpi=200, bbox_inches="tight")
    plt.savefig(f"{out}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}.pdf")
    print(f"Saved: {out}.png")


if __name__ == "__main__":
    main()
