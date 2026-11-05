#!/usr/bin/env python3
"""
Calibration plot: balanced vs trainagreeing (text + vision, trainval setting).
2×2 grid: top row = text, bottom row = vision.
Each row: (a) per-bin reliability, (b) cumulative accuracy.

Best checkpoints (test accuracy):
  text balanced:      ckpt-845   (66.5%)
  text trainagreeing: ckpt-1468  (65.9%)
  vision balanced:    ckpt-1692  (66.0%)
  vision trainagreeing: ckpt-1470 (65.3%)

Predictions are joined with test metadata via submission_id for correctness.
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

RUNS = {
    "text_balanced": {
        "pred_path": ROOT / "results/final_sweep_v7_datasweepv3/optim_search_trainval"
                          / "bz16_lr1e-6_text_balanced/finetuned-ckpt-845.jsonl",
        "dataset": "iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered",
    },
    "text_trainagreeing": {
        "pred_path": ROOT / "results/final_sweep_v7_datasweepv3/optim_search_trainval"
                          / "bz16_lr1e-6_text_trainagreeing/finetuned-ckpt-1468.jsonl",
        "dataset": "iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered",
    },
    "vision_balanced": {
        "pred_path": ROOT / "results/final_sweep_v7_datasweepv3/optim_search_trainval"
                          / "bz16_lr1e-6_vision_balanced/finetuned-ckpt-1692.jsonl",
        "dataset": "iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_labelfix_v7_filtered_filtered24480",
    },
    "vision_trainagreeing": {
        "pred_path": ROOT / "results/final_sweep_v7_datasweepv3/optim_search_trainval"
                          / "bz16_lr1e-6_vision_trainagreeing/finetuned-ckpt-1470.jsonl",
        "dataset": "iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_labelfix_v7_filtered_filtered24480",
    },
}

DECISION_TOKEN_IDX = 5
COLOR_BAL = "#6098FF"      # blue
COLOR_TA = "#FF8C69"       # salmon/orange-red
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


def load_predictions(pred_path: Path, dataset_name: str):
    """Load predictions and join with test metadata via submission_id."""
    # Load predictions
    preds = []
    with open(pred_path) as f:
        for line in f:
            preds.append(json.loads(line))

    # Load test dataset metadata — build submission_id → metadata map
    test_path = ROOT / "data" / f"{dataset_name}_test" / "data.json"
    if test_path.exists():
        test_data = json.load(open(test_path))
        meta_by_id = {}
        for ex in test_data:
            meta = ex.get("_metadata", {})
            sid = meta.get("submission_id")
            if sid:
                meta_by_id[sid] = meta
    else:
        meta_by_id = {}

    confidences, correct = [], []
    for rec in preds:
        pred = extract_prediction(rec["predict"])
        label = extract_prediction(rec["label"])
        conf = math.exp(rec["token_logprobs"][DECISION_TOKEN_IDX])
        confidences.append(conf)
        correct.append(pred == label and pred != "unknown")

    return np.array(confidences), np.array(correct)


def panel_perbin(ax, bal_conf, bal_corr, ta_conf, ta_corr):
    bins = np.linspace(0.5, 1.0, 11)
    for conf, corr, color in [
        (bal_conf, bal_corr, COLOR_BAL),
        (ta_conf, ta_corr, COLOR_TA),
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

    ax2 = ax.twinx()
    thresholds = np.linspace(0.5, 0.99, 200)
    for conf, color in [(bal_conf, COLOR_BAL), (ta_conf, COLOR_TA)]:
        coverages = [(conf >= t).mean() for t in thresholds]
        ax2.plot(thresholds, coverages, color=color, linestyle="--",
                 linewidth=LINEWIDTH - 0.5, alpha=0.6)
    ax2.set_ylabel("Coverage", fontsize=labelsize, color="#666666")
    ax2.set_ylim(0, 1.05)
    ax2.tick_params(axis="y", labelsize=ticksize, colors="#666666")

    for t_val in [0.7, 0.8, 0.9]:
        ax.axvline(x=t_val, color="#BBBBBB", linewidth=1.0, linestyle=":",
                   alpha=0.6, zorder=0)


def panel_cumulative(ax, bal_conf, bal_corr, ta_conf, ta_corr):
    thresholds = np.linspace(0.5, 0.98, 100)

    for conf, corr, color in [
        (bal_conf, bal_corr, COLOR_BAL),
        (ta_conf, ta_corr, COLOR_TA),
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

    for conf, corr, color, name, offset in [
        (ta_conf, ta_corr, COLOR_TA, "Trainagreeing", 12),
        (bal_conf, bal_corr, COLOR_BAL, "Balanced", -18),
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
    ax.set_ylabel("Accuracy (>= threshold)", fontsize=labelsize)
    ax.set_xlim(0.48, 1.02)
    ax.set_ylim(0.45, 1.05)
    ax.tick_params(axis="both", labelsize=ticksize)
    ax.grid(True, linestyle="--")

    ax2 = ax.twinx()
    for conf, color in [(bal_conf, COLOR_BAL), (ta_conf, COLOR_TA)]:
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
    # Load all four runs
    data = {}
    for name, cfg in RUNS.items():
        conf, corr = load_predictions(cfg["pred_path"], cfg["dataset"])
        data[name] = (conf, corr)
        print(f"{name}: {corr.mean():.4f} ({len(corr)} samples)")

    # 2×2 grid: top = text, bottom = vision
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    # Shared legend at top
    legend_handles = [
        Line2D([0], [0], color=COLOR_BAL, marker="o", linewidth=LINEWIDTH,
               markersize=MARKERSIZE, label="Balanced (acc.)"),
        Line2D([0], [0], color=COLOR_TA, marker="o", linewidth=LINEWIDTH,
               markersize=MARKERSIZE, label="Trainagreeing (acc.)"),
        Line2D([0], [0], color="grey", linewidth=1.5, linestyle="-",
               label="y = x"),
        Line2D([0], [0], color="#888888", linewidth=LINEWIDTH - 0.5,
               linestyle="--", label="Coverage"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=4,
               fontsize=legendsize, frameon=False, bbox_to_anchor=(0.5, 1.01))

    # Top row: Text
    tb, tc = data["text_balanced"]
    ttb, ttc = data["text_trainagreeing"]
    panel_perbin(axes[0, 0], tb, tc, ttb, ttc)
    panel_cumulative(axes[0, 1], tb, tc, ttb, ttc)
    axes[0, 0].set_title("(a) Text: Per-Bin Reliability", fontsize=titlesize, fontweight="bold", pad=15)
    axes[0, 1].set_title("(b) Text: Cumulative Accuracy", fontsize=titlesize, fontweight="bold", pad=15)

    # Bottom row: Vision
    vb, vc = data["vision_balanced"]
    vtb, vtc = data["vision_trainagreeing"]
    panel_perbin(axes[1, 0], vb, vc, vtb, vtc)
    panel_cumulative(axes[1, 1], vb, vc, vtb, vtc)
    axes[1, 0].set_title("(c) Vision: Per-Bin Reliability", fontsize=titlesize, fontweight="bold", pad=15)
    axes[1, 1].set_title("(d) Vision: Cumulative Accuracy", fontsize=titlesize, fontweight="bold", pad=15)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out = OUTPUT_DIR / "calibration_balanced_vs_trainagreeing"
    plt.savefig(f"{out}.pdf", dpi=200, bbox_inches="tight")
    plt.savefig(f"{out}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}.pdf")
    print(f"Saved: {out}.png")


if __name__ == "__main__":
    main()
