#!/usr/bin/env python3
"""
Reliability calibration plot comparing text training accept/reject ratios:
50/50, 40/60, 30/70.

Raw model confidence (no Platt scaling) on the 2025+2026 test set.
Uses the best 2025+2026 checkpoint for each ratio.

Output: calibration_ratio_sweep.pdf / .png
"""

import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

# ---------------------------------------------------------------------------
# Style guide (matches existing calibration scripts)
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
})

labelsize = 20
titlesize = 20
legendsize = 16
ticksize = 16

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "tmp_latex_dir" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DECISION_TOKEN_IDX = 5

# Each config: label, checkpoint path, metadata path, color
CONFIGS = [
    {
        "label": "50/50",
        "pred": ROOT / "results/final_sweep_v7_datasweepv3/optim_search_2026/bz32_lr1e-6_text/finetuned-ckpt-1322.jsonl",
        "meta": ROOT / "data/iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_v7_filtered_test/data.json",
        "color": "#6098FF",  # blue
    },
    {
        "label": "40/60",
        "pred": ROOT / "results/final_sweep_v7_datasweepv3/optim_search_2026/ratio_sweep/bz32_lr1e-6_text_40_60/finetuned-ckpt-1983.jsonl",
        "meta": ROOT / "data/iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_test/data.json",
        "color": "#FECC81",  # orange
    },
    {
        "label": "30/70",
        "pred": ROOT / "results/final_sweep_v7_datasweepv3/optim_search_2026/ratio_sweep/bz32_lr1e-6_text_30_70/finetuned-ckpt-1322.jsonl",
        "meta": ROOT / "data/iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_test/data.json",
        "color": "#7FB069",  # green
    },
]

LINEWIDTH = 3.0
MARKERSIZE = 10
EVAL_YEARS = {2025, 2026}


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


def load_predictions(pred_path: Path, meta_path: Path, keep_years: set):
    with open(meta_path) as f:
        metadata = [d["_metadata"] for d in json.load(f)]

    confidences, correct = [], []
    with open(pred_path) as f:
        for i, line in enumerate(f):
            rec = json.loads(line)
            year = metadata[i].get("year") if i < len(metadata) else None
            if year not in keep_years:
                continue
            pred = extract_prediction(rec["predict"])
            label = extract_prediction(rec["label"])
            if pred == "unknown" or label == "unknown":
                continue
            token_lps = rec.get("token_logprobs")
            if not token_lps or len(token_lps) <= DECISION_TOKEN_IDX:
                continue
            lp = token_lps[DECISION_TOKEN_IDX]
            if lp is None:
                continue
            conf = math.exp(lp)
            conf = np.clip(conf, 1e-7, 1 - 1e-7)
            confidences.append(conf)
            correct.append(pred == label)
    return np.array(confidences), np.array(correct, dtype=bool)


def panel_reliability(ax, results):
    bins = np.linspace(0.5, 1.0, 11)
    for res in results:
        conf = res["conf"]
        corr = res["corr"]
        accs, mean_confs = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (conf >= lo) & (conf < hi) if hi < 1.0 else (conf >= lo) & (conf <= hi)
            if mask.sum() > 0:
                accs.append(corr[mask].mean())
                mean_confs.append(conf[mask].mean())
            else:
                accs.append(np.nan)
                mean_confs.append(np.nan)
        ax.plot(
            mean_confs, accs, "o-",
            color=res["color"], linewidth=LINEWIDTH, markersize=MARKERSIZE,
            alpha=0.9, label=res["label"],
        )

    ax.plot([0.5, 1.0], [0.5, 1.0], "-", color="grey", linewidth=1.5)
    ax.set_xlabel("Binned Confidence", fontsize=labelsize)
    ax.set_ylabel("Accuracy", fontsize=labelsize)
    ax.set_xlim(0.48, 1.02)
    ax.set_ylim(0.45, 1.05)
    ax.tick_params(axis="both", labelsize=ticksize)
    ax.grid(False)

    # Secondary axis: cumulative coverage
    ax2 = ax.twinx()
    thresholds = np.linspace(0.5, 0.99, 200)
    for res in results:
        coverages = [(res["conf"] >= t).mean() for t in thresholds]
        ax2.plot(
            thresholds, coverages,
            color=res["color"], linestyle="--",
            linewidth=LINEWIDTH - 0.5, alpha=0.6,
        )
    ax2.set_ylabel("Cumulative Coverage", fontsize=labelsize, color="#666666")
    ax2.set_ylim(0, 1.05)
    ax2.tick_params(axis="y", labelsize=ticksize, colors="#666666")

    for t_val in [0.7, 0.8, 0.9]:
        ax.axvline(x=t_val, color="#BBBBBB", linewidth=1.0, linestyle=":",
                   alpha=0.6, zorder=0)

    coverage_handle = Line2D(
        [0], [0], color="#888888", linewidth=LINEWIDTH - 0.5,
        linestyle="--", label="Coverage",
    )
    handles, labels = ax.get_legend_handles_labels()
    handles.append(coverage_handle)
    labels.append("Coverage")
    ax.legend(handles, labels, fontsize=legendsize, loc="upper center",
              framealpha=0.95, ncol=4)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)


def panel_accuracy_by_pct_rating(ax, results, meta_map):
    """Bar chart: accuracy across [0,0.4], (0.4,0.6], (0.6,1.0] buckets."""
    bucket_edges = [(0.0, 0.4, "low\n[0, 0.4]"),
                    (0.4, 0.6, "mid\n(0.4, 0.6]"),
                    (0.6, 1.0, "high\n(0.6, 1.0]")]
    n_buckets = len(bucket_edges)
    n_models = len(results)
    x = np.arange(n_buckets)
    bar_width = 0.25

    for mi, res in enumerate(results):
        conf = res["conf"]
        corr = res["corr"]
        pct = np.array(res["pct"])

        accs = []
        ns = []
        for lo, hi, _ in bucket_edges:
            if lo == 0.0:
                mask = pct <= hi
            elif hi == 1.0:
                mask = (pct > lo) & (pct <= hi)
            else:
                mask = (pct > lo) & (pct <= hi)
            if mask.sum() > 0:
                accs.append(corr[mask].mean() * 100)
                ns.append(int(mask.sum()))
            else:
                accs.append(np.nan)
                ns.append(0)

        offset = (mi - (n_models - 1) / 2) * bar_width
        bars = ax.bar(
            x + offset, accs, bar_width,
            label=res["label"], color=res["color"],
            edgecolor="black", linewidth=0.8, alpha=0.9,
        )
        for bar, acc, n in zip(bars, accs, ns):
            if not np.isnan(acc):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.6,
                    f"{acc:.1f}",
                    ha="center", va="bottom", fontsize=ticksize - 2,
                )

    ax.axhline(y=50, color="grey", linewidth=1.0, linestyle=":", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([b[2] for b in bucket_edges], fontsize=ticksize)
    ax.set_ylabel("Accuracy (%)", fontsize=labelsize)
    ax.set_xlabel("Pctl. Rating Bucket", fontsize=labelsize)
    ax.set_ylim(45, 82)
    ax.tick_params(axis="y", labelsize=ticksize)
    ax.legend(fontsize=legendsize, loc="upper left", ncol=3, framealpha=0.95)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)


def main():
    # Load predictions + metadata for all three ratios
    results = []
    for cfg in CONFIGS:
        with open(cfg["meta"]) as f:
            metadata = [d["_metadata"] for d in json.load(f)]

        confidences, correct, pct_ratings = [], [], []
        with open(cfg["pred"]) as f:
            for i, line in enumerate(f):
                rec = json.loads(line)
                if i >= len(metadata):
                    break
                meta = metadata[i]
                year = meta.get("year")
                if year not in EVAL_YEARS:
                    continue
                pct = meta.get("pct_rating")
                pred = extract_prediction(rec["predict"])
                label = extract_prediction(rec["label"])
                if pred == "unknown" or label == "unknown" or pct is None:
                    continue
                token_lps = rec.get("token_logprobs")
                if not token_lps or len(token_lps) <= DECISION_TOKEN_IDX:
                    continue
                lp = token_lps[DECISION_TOKEN_IDX]
                if lp is None:
                    continue
                conf = math.exp(lp)
                conf = float(np.clip(conf, 1e-7, 1 - 1e-7))
                confidences.append(conf)
                correct.append(pred == label)
                pct_ratings.append(pct)

        results.append({
            "label": cfg["label"],
            "color": cfg["color"],
            "conf": np.array(confidences),
            "corr": np.array(correct, dtype=bool),
            "pct": pct_ratings,
        })
        acc = np.array(correct).mean() * 100
        print(f"{cfg['label']}: n={len(confidences)}  acc={acc:.1f}%  mean_conf={np.mean(confidences)*100:.1f}%")

    # Figure: 1x2 layout — reliability + per-bucket bars
    fig = plt.figure(figsize=(20, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1], wspace=0.35)

    ax_a = fig.add_subplot(gs[0, 0])
    panel_reliability(ax_a, results)

    ax_b = fig.add_subplot(gs[0, 1])
    panel_accuracy_by_pct_rating(ax_b, results, None)

    # Titles
    fig.canvas.draw()
    title_y = 0.97
    for ax_obj, label in [
        (ax_a, "(a) Reliability & Coverage ('25-'26, raw conf.)"),
        (ax_b, "(b) Accuracy by Pctl. Rating Bucket"),
    ]:
        pos = ax_obj.get_position()
        fig.text((pos.x0 + pos.x1) / 2, title_y, label,
                 ha="center", va="bottom", fontsize=titlesize)

    out = OUTPUT_DIR / "calibration_ratio_sweep"
    plt.savefig(f"{out}.pdf", dpi=200, bbox_inches="tight", pad_inches=0.2)
    plt.savefig(f"{out}.png", dpi=150, bbox_inches="tight", pad_inches=0.2)
    plt.close()
    print(f"\nSaved: {out}.pdf / .png")


if __name__ == "__main__":
    main()
