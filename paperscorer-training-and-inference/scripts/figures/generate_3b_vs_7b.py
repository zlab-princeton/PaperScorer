#!/usr/bin/env python3
"""
3B vs 7B comparison for both modalities (text and vision).

Layout: 2 rows (text, vision) x 5 cols:
    1. Train Loss (log scale)
    2. Train Acc on a 2k slice of train (from train-ckpt JSONs)
    3. Test Acc on ICLR 2025+26
    4. Accept Recall on 25+26
    5. Reject Recall on 25+26

Each panel has two lines: 3B (light) and 7B (dark), epoch on x-axis.

Output: tmp_latex_dir/figures/3b_vs_7b.{pdf,png}
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
})
labelsize = 18
titlesize = 18
legendsize = 14
ticksize = 14
LINEWIDTH = 2.4
MARKERSIZE = 9

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "tmp_latex_dir" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SAVES = ROOT / "saves/final_sweep_v7_datasweepv3/optim_search_2026"
RES = ROOT / "results/final_sweep_v7_datasweepv3/optim_search_2026"
DATA = ROOT / "data"

# (modality, size, save_dir, results_dir, ckpts, test_data)
TEXT_TEST_LABELFIX = DATA / "iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_test/data.json"
TEXT_TEST_NOLF    = DATA / "iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_v7_filtered_test/data.json"
VIS_TEST_LABELFIX = DATA / "iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_labelfix_v7_filtered_filtered24480_test/data.json"
VIS_TEST_NOLF     = DATA / "iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_v7_filtered_test/data.json"

CONFIGS = {
    "text": [
        ("3B",  SAVES / "scaling/bz32_lr1e-6_text_3b",
                RES   / "scaling/bz32_lr1e-6_text_3b",
                [661, 1322],
                [TEXT_TEST_LABELFIX, TEXT_TEST_NOLF]),
        ("7B",  SAVES / "bz32_lr1e-6_text",
                RES   / "bz32_lr1e-6_text",
                [661, 1322],
                [TEXT_TEST_NOLF, TEXT_TEST_LABELFIX]),
        ("14B", SAVES / "scaling/bz32_lr1e-6_text_14b",
                RES   / "scaling/bz32_lr1e-6_text_14b",
                [661, 1322],
                [TEXT_TEST_LABELFIX, TEXT_TEST_NOLF]),
    ],
    "vision": [
        ("3B", SAVES / "scaling/bz16_lr1e-6_vision_3b",
               RES   / "scaling/bz16_lr1e-6_vision_3b",
               [1324, 2648],
               [VIS_TEST_LABELFIX, VIS_TEST_NOLF]),
        ("7B", SAVES / "bz16_lr1e-6_vision",
               RES   / "bz16_lr1e-6_vision",
               [1324, 2648],
               [VIS_TEST_LABELFIX, VIS_TEST_NOLF]),
    ],
}

# X-axis epoch cap (matching the 2-epoch scope above)
EPOCH_CAP = 2.05

EVAL_YEARS = {2025, 2026}

# Colors per (modality, size) — pairs/triples that are visually distinct
# Text uses a blues ramp, vision uses an oranges ramp.
COLOR = {
    ("text", "3B"):    "#9ECAE1",
    ("text", "7B"):    "#3182BD",
    ("text", "14B"):   "#08306B",
    ("vision", "3B"):  "#FDAE6B",
    ("vision", "7B"):  "#A63603",
}


def extract_pred(text):
    t = text.lower().strip()
    if "\\boxed{accept}" in t or "boxed{accept}" in t: return "accept"
    if "\\boxed{reject}" in t or "boxed{reject}" in t: return "reject"
    if "accept" in t: return "accept"
    if "reject" in t: return "reject"
    return "unknown"


def load_train_loss(save_dir):
    """Return arrays of (steps, epochs, losses)."""
    log = save_dir / "trainer_log.jsonl"
    steps, epochs, losses = [], [], []
    if not log.exists():
        return np.array(steps), np.array(epochs), np.array(losses)
    with log.open() as f:
        for line in f:
            try: r = json.loads(line)
            except Exception: continue
            if "loss" not in r: continue
            steps.append(r["current_steps"])
            epochs.append(r.get("epoch", 0.0))
            losses.append(r["loss"])
    return np.array(steps), np.array(epochs), np.array(losses)


def smooth(y, window=10):
    if len(y) < window: return y
    k = np.ones(window) / window
    return np.convolve(y, k, mode="valid")


def load_train_acc(results_dir, step):
    """Return eval_sft_accuracy from train-ckpt-{step}.json (None if missing)."""
    p = results_dir / f"train-ckpt-{step}.json"
    if not p.exists():
        return None
    try:
        d = json.load(open(p))
        return d.get("eval_sft_accuracy") or d.get("sft_accuracy")
    except Exception:
        return None


def test_metrics_2526(jsonl, test_data_paths):
    """Return (acc, accr, rejr, n) on 25/26. Tries each test_data_paths in order."""
    if not jsonl.exists():
        return (None, None, None, 0)
    test_meta = None
    for p in test_data_paths:
        if p.exists():
            test_meta = json.loads(p.read_text())
            break
    if test_meta is None:
        return (None, None, None, 0)

    tp = tn = fp = fn = 0
    with jsonl.open() as f:
        for i, line in enumerate(f):
            if i >= len(test_meta): break
            yr = (test_meta[i].get("_metadata") or {}).get("year")
            if yr not in EVAL_YEARS: continue
            r = json.loads(line)
            p_ = extract_pred(r.get("predict", ""))
            g_ = extract_pred(r.get("label", ""))
            if p_ == "unknown" or g_ == "unknown": continue
            if p_ == "accept" and g_ == "accept": tp += 1
            elif p_ == "reject" and g_ == "reject": tn += 1
            elif p_ == "accept" and g_ == "reject": fp += 1
            elif p_ == "reject" and g_ == "accept": fn += 1

    n = tp + tn + fp + fn
    if n == 0:
        return (None, None, None, 0)
    acc = (tp + tn) / n * 100
    accr = tp / (tp + fn) * 100 if (tp + fn) else None
    rejr = tn / (tn + fp) * 100 if (tn + fp) else None
    return (acc, accr, rejr, n)


def epoch_for_step(epochs, steps, target):
    if len(steps) == 0: return 0.0
    idx = int(np.argmin(np.abs(steps - target)))
    return float(epochs[idx])


def style_axis(ax, title, ylabel):
    ax.set_xlabel("Epoch", fontsize=labelsize - 2)
    ax.set_ylabel(ylabel, fontsize=labelsize)
    ax.set_title(title, fontsize=titlesize - 2, pad=8)
    ax.tick_params(axis="both", labelsize=ticksize)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(1.1)


def plot_loss(ax, modality):
    all_losses: list[float] = []
    for size, sd, _, _, _ in CONFIGS[modality]:
        steps, epochs, losses = load_train_loss(sd)
        if len(losses) == 0: continue
        # Clip to first EPOCH_CAP epochs
        mask = epochs <= EPOCH_CAP
        epochs = epochs[mask]
        losses = losses[mask]
        if len(losses) == 0: continue
        c = COLOR[(modality, size)]
        ax.plot(epochs, losses, color=c, linewidth=0.8, alpha=0.25)
        if len(losses) >= 10:
            ys = smooth(losses, 10)
            offset = (len(losses) - len(ys)) // 2
            xs = epochs[offset:offset + len(ys)]
            ax.plot(xs, ys, color=c, linewidth=LINEWIDTH, label=f"{size}")
            all_losses.extend(ys.tolist())
        else:
            ax.plot(epochs, losses, color=c, linewidth=LINEWIDTH, label=f"{size}")
            all_losses.extend(losses.tolist())
    ax.set_yscale("log")
    if all_losses:
        positives = [v for v in all_losses if v > 0]
        if positives:
            lo = min(positives) * 0.85
            ax.set_ylim(lo, 1.0)
    ax.set_xlim(0, EPOCH_CAP)
    style_axis(ax, f"{modality.title()} — Train Loss", "Loss (log)")


def plot_train_acc(ax, modality):
    all_y = []
    for size, sd, rd, ckpts, _ in CONFIGS[modality]:
        steps, epochs, losses = load_train_loss(sd)
        xs, ys = [], []
        for c in ckpts:
            v = load_train_acc(rd, c)
            if v is None: continue
            xs.append(epoch_for_step(epochs, steps, c))
            ys.append(v * 100)  # eval_sft_accuracy is in [0,1]
        if not xs: continue
        all_y.extend(ys)
        col = COLOR[(modality, size)]
        ax.plot(xs, ys, "-o", color=col, linewidth=LINEWIDTH,
                markersize=MARKERSIZE, label=f"{size}")
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.1f}", (x, y), xytext=(0, 8),
                        textcoords="offset points", ha="center",
                        fontsize=ticksize - 4, color=col, fontweight="bold")
    if all_y:
        lo, hi = min(all_y), max(all_y); span = max(hi - lo, 2)
        ax.set_ylim(lo - 0.15 * span, hi + 0.4 * span)
    style_axis(ax, f"{modality.title()} — Train Acc (2k slice)", "Accuracy (%)")


def plot_test_metric(ax, modality, key, ylabel, title_metric):
    """key: 'acc' / 'accr' / 'rejr' (index 0/1/2 in test_metrics_2526 tuple)."""
    idx_map = {"acc": 0, "accr": 1, "rejr": 2}
    idx = idx_map[key]
    all_y = []
    for size, sd, rd, ckpts, td in CONFIGS[modality]:
        steps, epochs, losses = load_train_loss(sd)
        xs, ys = [], []
        for c in ckpts:
            jp = rd / f"finetuned-ckpt-{c}.jsonl"
            metrics = test_metrics_2526(jp, td)
            v = metrics[idx]
            if v is None: continue
            xs.append(epoch_for_step(epochs, steps, c))
            ys.append(v)
        if not xs: continue
        all_y.extend(ys)
        col = COLOR[(modality, size)]
        ax.plot(xs, ys, "-o", color=col, linewidth=LINEWIDTH,
                markersize=MARKERSIZE, label=f"{size}")
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.1f}", (x, y), xytext=(0, 8),
                        textcoords="offset points", ha="center",
                        fontsize=ticksize - 4, color=col, fontweight="bold")
    if all_y:
        lo, hi = min(all_y), max(all_y); span = max(hi - lo, 4)
        ax.set_ylim(lo - 0.15 * span, hi + 0.4 * span)
    style_axis(ax, f"{modality.title()} — {title_metric}", ylabel)


def main():
    fig, axes = plt.subplots(2, 5, figsize=(28, 9))

    for row, modality in enumerate(["text", "vision"]):
        plot_loss(axes[row, 0], modality)
        plot_train_acc(axes[row, 1], modality)
        plot_test_metric(axes[row, 2], modality, "acc", "Accuracy (%)",
                         "Test Acc (25/26)")
        plot_test_metric(axes[row, 3], modality, "accr", "Accept Recall (%)",
                         "Accept Recall (25/26)")
        plot_test_metric(axes[row, 4], modality, "rejr", "Reject Recall (%)",
                         "Reject Recall (25/26)")

    # One legend per row, in upper-right of col-0 axes
    for row, modality in enumerate(["text", "vision"]):
        axes[row, 0].legend(fontsize=legendsize, loc="upper right",
                            framealpha=0.95)

    fig.suptitle("Model Scaling — Text (3B/7B/14B) and Vision (3B/7B) over First 2 Epochs",
                 fontsize=titlesize + 4, fontweight="bold", y=1.005)
    plt.tight_layout()

    out = OUT_DIR / "3b_vs_7b"
    plt.savefig(f"{out}.pdf", dpi=200, bbox_inches="tight", pad_inches=0.2)
    plt.savefig(f"{out}.png", dpi=150, bbox_inches="tight", pad_inches=0.2)
    plt.close()
    print(f"Saved: {out}.pdf / .png")

    # Print numeric summary
    print("\n=== Summary (epoch-2 ckpt: text=1322, vision=2648) ===")
    for modality in ("text", "vision"):
        epoch2 = 1322 if modality == "text" else 2648
        for size, sd, rd, ckpts, td in CONFIGS[modality]:
            jp = rd / f"finetuned-ckpt-{epoch2}.jsonl"
            ta = load_train_acc(rd, epoch2)
            ta_pct = ta * 100 if ta is not None else None
            acc, accr, rejr, n = test_metrics_2526(jp, td)
            ta_s = f"{ta_pct:.1f}%" if ta_pct is not None else " N/A "
            acc_s = f"{acc:.1f}%" if acc is not None else " N/A"
            accr_s = f"{accr:.1f}%" if accr is not None else " N/A"
            rejr_s = f"{rejr:.1f}%" if rejr is not None else " N/A"
            print(f"  {modality:<7} {size:<4}: trainAcc={ta_s:>7}  testAcc={acc_s:>6}  "
                  f"AccR={accr_s:>6}  RejR={rejr_s:>6}  (n={n})")


if __name__ == "__main__":
    main()
