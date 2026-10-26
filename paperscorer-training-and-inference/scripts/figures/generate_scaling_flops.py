#!/usr/bin/env python3
"""
Scaling-laws plot: metrics vs training FLOPs for text (3B/7B/14B) and
vision (3B/7B). Standard Chinchilla approximation:

    FLOPs ≈ 6 · N · D

where N is total trainable params and D is total tokens seen at the
current checkpoint (D = step · effective_batch · cutoff_len).

Layout: 2 rows (text, vision) x 4 cols
    1. Train Loss  vs FLOPs  (every 10-step log entry, smoothed)
    2. Test Acc on 25/26 vs FLOPs (one point per saved checkpoint)
    3. Accept Recall on 25/26 vs FLOPs
    4. Reject Recall on 25/26 vs FLOPs

Each panel has one line per model size; points are the saved checkpoints.

Output: tmp_latex_dir/figures/scaling_flops.{pdf,png}
"""

from __future__ import annotations

import json
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
MARKERSIZE = 10

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "tmp_latex_dir" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SAVES = ROOT / "saves/final_sweep_v7_datasweepv3/optim_search_2026"
RES = ROOT / "results/final_sweep_v7_datasweepv3/optim_search_2026"
DATA = ROOT / "data"

TEXT_TEST_LABELFIX = DATA / "iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_test/data.json"
TEXT_TEST_NOLF    = DATA / "iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_v7_filtered_test/data.json"
VIS_TEST_LABELFIX = DATA / "iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_labelfix_v7_filtered_filtered24480_test/data.json"
VIS_TEST_NOLF     = DATA / "iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_v7_filtered_test/data.json"

# Parameter counts (total parameters, including frozen vision tower for VL models)
PARAMS = {
    ("text", "3B"):  3.09e9,
    ("text", "7B"):  7.62e9,
    ("text", "14B"): 14.77e9,
    ("vision", "3B"): 3.75e9,
    ("vision", "7B"): 8.29e9,
}

# Effective batch (per_device * num_gpus * grad_accum)
EFFECTIVE_BATCH = {"text": 32, "vision": 16}
CUTOFF_LEN = 24480  # also our nominal tokens-per-sample (upper bound)

CONFIGS = {
    "text": [
        ("3B",  SAVES / "scaling/bz32_lr1e-6_text_3b",
                RES   / "scaling/bz32_lr1e-6_text_3b",
                [661, 1322, 1983, 2644],
                [TEXT_TEST_LABELFIX, TEXT_TEST_NOLF]),
        ("7B",  SAVES / "bz32_lr1e-6_text",
                RES   / "bz32_lr1e-6_text",
                [661, 1322, 1983, 2644],
                [TEXT_TEST_NOLF, TEXT_TEST_LABELFIX]),
        ("14B", SAVES / "scaling/bz32_lr1e-6_text_14b",
                RES   / "scaling/bz32_lr1e-6_text_14b",
                [661, 1322],
                [TEXT_TEST_LABELFIX, TEXT_TEST_NOLF]),
    ],
    "vision": [
        ("3B", SAVES / "scaling/bz16_lr1e-6_vision_3b",
               RES   / "scaling/bz16_lr1e-6_vision_3b",
               [1324, 2648, 3972, 5296],
               [VIS_TEST_LABELFIX, VIS_TEST_NOLF]),
        ("7B", SAVES / "bz16_lr1e-6_vision",
               RES   / "bz16_lr1e-6_vision",
               [1324, 2648, 3972, 5296],
               [VIS_TEST_LABELFIX, VIS_TEST_NOLF]),
    ],
}

EVAL_YEARS = {2025, 2026}
COLOR = {
    ("text", "3B"):    "#9ECAE1",
    ("text", "7B"):    "#3182BD",
    ("text", "14B"):   "#08306B",
    ("vision", "3B"):  "#FDAE6B",
    ("vision", "7B"):  "#A63603",
}


def flops_at_step(modality: str, size: str, step: int) -> float:
    """Total training FLOPs at a given step. 6 * N * D (Kaplan/Chinchilla)."""
    n = PARAMS[(modality, size)]
    batch = EFFECTIVE_BATCH[modality]
    tokens = step * batch * CUTOFF_LEN
    return 6.0 * n * tokens


def extract_pred(text):
    t = text.lower().strip()
    if "\\boxed{accept}" in t or "boxed{accept}" in t: return "accept"
    if "\\boxed{reject}" in t or "boxed{reject}" in t: return "reject"
    if "accept" in t: return "accept"
    if "reject" in t: return "reject"
    return "unknown"


def load_train_loss(save_dir):
    log = save_dir / "trainer_log.jsonl"
    steps, losses = [], []
    if not log.exists():
        return np.array(steps), np.array(losses)
    with log.open() as f:
        for line in f:
            try: r = json.loads(line)
            except Exception: continue
            if "loss" not in r: continue
            steps.append(r["current_steps"])
            losses.append(r["loss"])
    return np.array(steps), np.array(losses)


def smooth(y, window=10):
    if len(y) < window: return y
    k = np.ones(window) / window
    return np.convolve(y, k, mode="valid")


def test_metrics_2526(jsonl, test_data_paths):
    if not jsonl.exists(): return (None, None, None, 0)
    test_meta = None
    for p in test_data_paths:
        if p.exists():
            test_meta = json.loads(p.read_text()); break
    if test_meta is None: return (None, None, None, 0)
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
    if n == 0: return (None, None, None, 0)
    acc = (tp + tn) / n * 100
    accr = tp / (tp + fn) * 100 if (tp + fn) else None
    rejr = tn / (tn + fp) * 100 if (tn + fp) else None
    return (acc, accr, rejr, n)


def style_axis(ax, title, ylabel):
    ax.set_xlabel("Training FLOPs (log)", fontsize=labelsize - 2)
    ax.set_ylabel(ylabel, fontsize=labelsize)
    ax.set_title(title, fontsize=titlesize - 2, pad=8)
    ax.set_xscale("log")
    ax.tick_params(axis="both", labelsize=ticksize)
    ax.grid(True, axis="both", linestyle=":", alpha=0.4)
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(1.1)


def plot_loss(ax, modality):
    all_losses: list[float] = []
    for size, sd, _, _, _ in CONFIGS[modality]:
        steps, losses = load_train_loss(sd)
        if len(losses) == 0: continue
        flops = np.array([flops_at_step(modality, size, int(s)) for s in steps])
        c = COLOR[(modality, size)]
        ax.plot(flops, losses, color=c, linewidth=0.8, alpha=0.25)
        if len(losses) >= 10:
            ys = smooth(losses, 10)
            offset = (len(losses) - len(ys)) // 2
            xs = flops[offset:offset + len(ys)]
            ax.plot(xs, ys, color=c, linewidth=LINEWIDTH, label=size)
            all_losses.extend(ys.tolist())
        else:
            ax.plot(flops, losses, color=c, linewidth=LINEWIDTH, label=size)
            all_losses.extend(losses.tolist())
    ax.set_yscale("log")
    if all_losses:
        positives = [v for v in all_losses if v > 0]
        if positives:
            ax.set_ylim(min(positives) * 0.85, 1.0)
    style_axis(ax, f"{modality.title()} — Train Loss", "Loss (log)")


def plot_test_metric(ax, modality, key, ylabel, title_metric):
    idx_map = {"acc": 0, "accr": 1, "rejr": 2}
    idx = idx_map[key]
    all_y = []
    for size, sd, rd, ckpts, td in CONFIGS[modality]:
        xs, ys = [], []
        for c in ckpts:
            jp = rd / f"finetuned-ckpt-{c}.jsonl"
            metrics = test_metrics_2526(jp, td)
            v = metrics[idx]
            if v is None: continue
            xs.append(flops_at_step(modality, size, c))
            ys.append(v)
        if not xs: continue
        all_y.extend(ys)
        col = COLOR[(modality, size)]
        ax.plot(xs, ys, "-o", color=col, linewidth=LINEWIDTH,
                markersize=MARKERSIZE, label=size)
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.1f}", (x, y), xytext=(0, 8),
                        textcoords="offset points", ha="center",
                        fontsize=ticksize - 4, color=col, fontweight="bold")
    if all_y:
        lo, hi = min(all_y), max(all_y); span = max(hi - lo, 4)
        ax.set_ylim(lo - 0.15 * span, hi + 0.4 * span)
    style_axis(ax, f"{modality.title()} — {title_metric}", ylabel)


def main():
    fig, axes = plt.subplots(2, 4, figsize=(24, 9))

    for row, modality in enumerate(["text", "vision"]):
        plot_loss(axes[row, 0], modality)
        plot_test_metric(axes[row, 1], modality, "acc",
                         "Accuracy (%)", "Test Acc (25/26)")
        plot_test_metric(axes[row, 2], modality, "accr",
                         "Accept Recall (%)", "Accept Recall (25/26)")
        plot_test_metric(axes[row, 3], modality, "rejr",
                         "Reject Recall (%)", "Reject Recall (25/26)")
        # Legend in upper-left of leftmost panel only
        axes[row, 0].legend(fontsize=legendsize, loc="upper right",
                            framealpha=0.95)

    fig.suptitle("Scaling Laws — Metrics vs Training FLOPs (6·N·D)",
                 fontsize=titlesize + 4, fontweight="bold", y=1.005)
    plt.tight_layout()

    out = OUT_DIR / "scaling_flops"
    plt.savefig(f"{out}.pdf", dpi=200, bbox_inches="tight", pad_inches=0.2)
    plt.savefig(f"{out}.png", dpi=150, bbox_inches="tight", pad_inches=0.2)
    plt.close()
    print(f"Saved: {out}.pdf / .png")

    # Print numeric summary at epoch-2
    print("\n=== Epoch-2 ckpt FLOPs and metrics ===")
    for modality in ("text", "vision"):
        epoch2 = 1322 if modality == "text" else 2648
        for size, sd, rd, ckpts, td in CONFIGS[modality]:
            jp = rd / f"finetuned-ckpt-{epoch2}.jsonl"
            f = flops_at_step(modality, size, epoch2)
            acc, accr, rejr, n = test_metrics_2526(jp, td)
            acc_s = f"{acc:.1f}%" if acc is not None else " N/A"
            accr_s = f"{accr:.1f}%" if accr is not None else " N/A"
            rejr_s = f"{rejr:.1f}%" if rejr is not None else " N/A"
            print(f"  {modality:<7} {size:<4}: {f:.2e} FLOPs ({f/1e18:.1f} EFLOPs)  "
                  f"acc={acc_s:>6}  AccR={accr_s:>6}  RejR={rejr_s:>6}")


if __name__ == "__main__":
    main()
