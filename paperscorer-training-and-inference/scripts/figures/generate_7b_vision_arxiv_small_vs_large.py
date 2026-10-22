#!/usr/bin/env python3
"""
7B vision arxiv-trained: small (21k balanced) vs large (71k balanced per-venue).

Layout: 2 rows x 5 cols
  Row 0: SMALL (arxiv_21k_vision)
  Row 1: LARGE (arxiv_balanced_per_venue_vision)

Cols:
  0. Train Loss (log scale, smoothed)
  1. ICLR 25/26 — Test Acc
  2. ICLR 25/26 — Accept Recall + Reject Recall (single panel, two lines)
  3. ARXIV y24up — Test Acc
  4. ARXIV y24up — Accept Recall + Reject Recall (single panel, two lines)

Eval JSONL files live under
  results/.../arxiv_train/{small,large}/<name>/<TAG>/finetuned-ckpt-<step>{,-gpu-test}.jsonl
where TAG is iclr_balanced_test or arxiv_balanced_test (already filtered to
y25up / y24up — no further year filtering needed).
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
legendsize = 13
ticksize = 13
LINEWIDTH = 2.4
MARKERSIZE = 9

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "tmp_latex_dir" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SAVES_BASE = ROOT / "saves/final_sweep_v7_datasweepv3/final_data_sweep_v3/arxiv_train"
RES_BASE   = ROOT / "results/final_sweep_v7_datasweepv3/final_data_sweep_v3/arxiv_train"

# (label, save_dir, results_dir, ckpts)
ROWS = [
    ("Small (arxiv-21k balanced, 7B)",
        SAVES_BASE / "small/arxiv_21k_vision",
        RES_BASE   / "small/arxiv_21k_vision",
        [1309, 2618, 3927, 5236]),
    ("Large (arxiv-71k balanced per-venue, 7B)",
        SAVES_BASE / "large/arxiv_balanced_per_venue_vision",
        RES_BASE   / "large/arxiv_balanced_per_venue_vision",
        [4431, 8862, 13293]),
]

# Color scheme (oranges for vision, distinct shade per row)
ROW_COLOR = {
    0: "#FDAE6B",  # small
    1: "#A63603",  # large
}
# Single-row plots use the row's color for "Accept" and a complementary darker
# tone for "Reject" so the two lines are easy to tell apart.
RECALL_COLORS = {
    "accept": "#2C7FB8",  # blue
    "reject": "#D7301F",  # red
}


def extract_pred(text):
    t = (text or "").lower().strip()
    if "boxed{accept}" in t: return "accept"
    if "boxed{reject}" in t: return "reject"
    if "accept" in t: return "accept"
    if "reject" in t: return "reject"
    return "unknown"


def load_train_loss(save_dir):
    log = save_dir / "trainer_log.jsonl"
    steps, epochs, losses = [], [], []
    if not log.exists():
        return np.array(steps), np.array(epochs), np.array(losses)
    with log.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if "loss" not in r:
                continue
            steps.append(r["current_steps"])
            epochs.append(r.get("epoch", 0.0))
            losses.append(r["loss"])
    return np.array(steps), np.array(epochs), np.array(losses)


def smooth(y, window=15):
    if len(y) < window:
        return y
    k = np.ones(window) / window
    return np.convolve(y, k, mode="valid")


def find_jsonl(results_dir, tag, step):
    base = results_dir / tag
    for fname in (f"finetuned-ckpt-{step}.jsonl",
                  f"finetuned-ckpt-{step}-gpu-test.jsonl"):
        f = base / fname
        if f.exists() and f.stat().st_size > 0:
            return f
    return None


def test_metrics(jsonl_path):
    """Compute (acc, accr, rejr, n) over the entire jsonl (no extra filtering)."""
    if jsonl_path is None or not jsonl_path.exists():
        return (None, None, None, 0)
    tp = tn = fp = fn = 0
    with jsonl_path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            p_ = extract_pred(r.get("predict", ""))
            g_ = extract_pred(r.get("label", ""))
            if p_ == "unknown" or g_ == "unknown":
                continue
            if   p_ == "accept" and g_ == "accept": tp += 1
            elif p_ == "reject" and g_ == "reject": tn += 1
            elif p_ == "accept" and g_ == "reject": fp += 1
            elif p_ == "reject" and g_ == "accept": fn += 1
    n = tp + tn + fp + fn
    if n == 0:
        return (None, None, None, 0)
    acc  = (tp + tn) / n * 100
    accr = tp / (tp + fn) * 100 if (tp + fn) else None
    rejr = tn / (tn + fp) * 100 if (tn + fp) else None
    return (acc, accr, rejr, n)


def epoch_for_step(epochs, steps, target):
    if len(steps) == 0:
        return float(target)
    idx = int(np.argmin(np.abs(steps - target)))
    return float(epochs[idx])


def style_axis(ax, title, ylabel):
    ax.set_xlabel("Epoch", fontsize=labelsize - 2)
    ax.set_ylabel(ylabel, fontsize=labelsize - 2)
    ax.set_title(title, fontsize=titlesize - 2, pad=8)
    ax.tick_params(axis="both", labelsize=ticksize)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(1.1)


def plot_loss(ax, save_dir, row_idx, row_label):
    steps, epochs, losses = load_train_loss(save_dir)
    if len(losses) == 0:
        ax.text(0.5, 0.5, "no trainer_log.jsonl", ha="center", va="center",
                transform=ax.transAxes, fontsize=14, color="gray")
        style_axis(ax, f"{row_label}\nTrain Loss", "Loss (log)")
        return
    c = ROW_COLOR[row_idx]
    ax.plot(epochs, losses, color=c, linewidth=0.7, alpha=0.25)
    if len(losses) >= 15:
        ys = smooth(losses, 15)
        offset = (len(losses) - len(ys)) // 2
        xs = epochs[offset:offset + len(ys)]
        ax.plot(xs, ys, color=c, linewidth=LINEWIDTH, label="loss (smoothed)")
    else:
        ax.plot(epochs, losses, color=c, linewidth=LINEWIDTH, label="loss")
    pos = [v for v in losses if v > 0]
    if pos:
        ax.set_yscale("log")
        ax.set_ylim(min(pos) * 0.85, max(pos) * 1.15)
    ax.set_xlim(0, max(epochs) if len(epochs) else 4.0)
    style_axis(ax, "Train Loss", "Loss (log)")
    ax.legend(fontsize=legendsize - 1, loc="upper right", framealpha=0.95)


def plot_acc(ax, results_dir, save_dir, ckpts, tag, row_idx, panel_title):
    steps, epochs, _ = load_train_loss(save_dir)
    xs, ys = [], []
    for c in ckpts:
        jp = find_jsonl(results_dir, tag, c)
        acc, _, _, n = test_metrics(jp)
        if acc is None:
            continue
        xs.append(epoch_for_step(epochs, steps, c))
        ys.append(acc)
    col = ROW_COLOR[row_idx]
    if xs:
        ax.plot(xs, ys, "-o", color=col, linewidth=LINEWIDTH,
                markersize=MARKERSIZE)
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.1f}", (x, y), xytext=(0, 8),
                        textcoords="offset points", ha="center",
                        fontsize=ticksize - 3, color=col, fontweight="bold")
        lo, hi = min(ys), max(ys); span = max(hi - lo, 4)
        ax.set_ylim(lo - 0.2 * span, hi + 0.4 * span)
    else:
        ax.text(0.5, 0.5, "no eval data", ha="center", va="center",
                transform=ax.transAxes, fontsize=14, color="gray")
    style_axis(ax, panel_title, "Accuracy (%)")


def plot_recall(ax, results_dir, save_dir, ckpts, tag, row_idx, panel_title):
    steps, epochs, _ = load_train_loss(save_dir)
    xs, accr_y, rejr_y = [], [], []
    for c in ckpts:
        jp = find_jsonl(results_dir, tag, c)
        _, accr, rejr, n = test_metrics(jp)
        if accr is None and rejr is None:
            continue
        xs.append(epoch_for_step(epochs, steps, c))
        accr_y.append(accr if accr is not None else np.nan)
        rejr_y.append(rejr if rejr is not None else np.nan)
    if xs:
        ax.plot(xs, accr_y, "-o", color=RECALL_COLORS["accept"],
                linewidth=LINEWIDTH, markersize=MARKERSIZE, label="Accept Recall")
        ax.plot(xs, rejr_y, "--s", color=RECALL_COLORS["reject"],
                linewidth=LINEWIDTH, markersize=MARKERSIZE, label="Reject Recall")
        for x, y in zip(xs, accr_y):
            if not np.isnan(y):
                ax.annotate(f"{y:.1f}", (x, y), xytext=(0, 8),
                            textcoords="offset points", ha="center",
                            fontsize=ticksize - 4,
                            color=RECALL_COLORS["accept"], fontweight="bold")
        for x, y in zip(xs, rejr_y):
            if not np.isnan(y):
                ax.annotate(f"{y:.1f}", (x, y), xytext=(0, -14),
                            textcoords="offset points", ha="center",
                            fontsize=ticksize - 4,
                            color=RECALL_COLORS["reject"], fontweight="bold")
        all_y = [v for v in accr_y + rejr_y if not np.isnan(v)]
        if all_y:
            lo, hi = min(all_y), max(all_y); span = max(hi - lo, 6)
            ax.set_ylim(max(0, lo - 0.2 * span), min(100, hi + 0.4 * span))
        ax.legend(fontsize=legendsize - 1, loc="best", framealpha=0.9)
    else:
        ax.text(0.5, 0.5, "no eval data", ha="center", va="center",
                transform=ax.transAxes, fontsize=14, color="gray")
    style_axis(ax, panel_title, "Recall (%)")


def main():
    fig, axes = plt.subplots(2, 5, figsize=(28, 9))

    for r, (label, sd, rd, ckpts) in enumerate(ROWS):
        plot_loss(axes[r, 0], sd, r, label)
        plot_acc   (axes[r, 1], rd, sd, ckpts, "iclr_balanced_test",  r,
                    "ICLR 25/26 — Test Acc")
        plot_recall(axes[r, 2], rd, sd, ckpts, "iclr_balanced_test",  r,
                    "ICLR 25/26 — Accept/Reject Recall")
        plot_acc   (axes[r, 3], rd, sd, ckpts, "arxiv_balanced_test", r,
                    "Arxiv y24+ — Test Acc")
        plot_recall(axes[r, 4], rd, sd, ckpts, "arxiv_balanced_test", r,
                    "Arxiv y24+ — Accept/Reject Recall")
        # Row label in left margin
        axes[r, 0].set_ylabel(f"{label}\n\nLoss (log)",
                              fontsize=labelsize - 4, fontweight="bold")

    fig.suptitle("7B Vision Arxiv-Trained — Small (21k) vs Large (71k) over Epochs",
                 fontsize=titlesize + 4, fontweight="bold", y=1.005)
    plt.tight_layout()

    out = OUT_DIR / "7b_vision_arxiv_small_vs_large"
    plt.savefig(f"{out}.pdf", dpi=200, bbox_inches="tight", pad_inches=0.2)
    plt.savefig(f"{out}.png", dpi=150, bbox_inches="tight", pad_inches=0.2)
    plt.close()
    print(f"Saved: {out}.pdf / .png")

    # Numeric summary
    print("\n=== Summary (per ckpt) ===")
    print(f"{'Row':<48} {'ckpt':>6} {'epoch':>6} "
          f"{'iclrAcc':>8} {'iclrAccR':>9} {'iclrRejR':>9} {'iclrN':>6} "
          f"{'arxAcc':>7} {'arxAccR':>8} {'arxRejR':>8} {'arxN':>6}")
    for label, sd, rd, ckpts in ROWS:
        steps, epochs, _ = load_train_loss(sd)
        for c in ckpts:
            ep = epoch_for_step(epochs, steps, c)
            i_acc, i_accr, i_rejr, i_n = test_metrics(find_jsonl(rd, "iclr_balanced_test", c))
            a_acc, a_accr, a_rejr, a_n = test_metrics(find_jsonl(rd, "arxiv_balanced_test", c))
            def s(v, w):
                return (f"{v:.1f}" if v is not None else "N/A").rjust(w)
            print(f"{label:<48} {c:>6} {ep:>6.2f} "
                  f"{s(i_acc,8)} {s(i_accr,9)} {s(i_rejr,9)} {i_n:>6} "
                  f"{s(a_acc,7)} {s(a_accr,8)} {s(a_rejr,8)} {a_n:>6}")


if __name__ == "__main__":
    main()
