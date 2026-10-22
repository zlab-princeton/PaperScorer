#!/usr/bin/env python3
"""
7B arxiv-trained: balanced vs natural-prior comparison, 4x4 grid.

Rows (modality x size):
  0. text  small  (21k)
  1. vision small (21k)
  2. text  large  (71k balanced per-venue / 47k natrate per-venue)
  3. vision large (71k balanced per-venue / 47k natrate per-venue)

Cols (eval test sets):
  0. arxiv balanced y24+
  1. arxiv natrate  y24+
  2. ICLR  balanced 25/26
  3. ICLR  natrate  25/26

Per cell (test accuracy over epochs):
  - Solid line  = balanced-trained model
  - Dashed line = natrate-trained model
  - For *natrate* eval cols, an extra dotted line per model shows
    accuracy under a *calibrated* threshold T* found on the matching
    natrate-VAL split by maximizing balanced accuracy.
    (Calibration is a no-op on balanced eval cols, so we omit it there.)

Calibration:
  - Per JSONL row, find decision-token position k = argmax_k |la[k]-lr[k]|
    among positions where both logprob_accept[k] and logprob_reject[k] are
    non-None.
  - P(accept) = exp(la[k]) / (exp(la[k]) + exp(lr[k]))  (2-class softmax)
  - On VAL split, sweep all unique P(accept) thresholds; pick T* maximizing
    balanced acc = (recall_accept + recall_reject) / 2.
  - Apply T* to TEST split: predict accept iff P(accept) > T*.

Skips cells with no data and prints a coverage summary.
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
labelsize = 16
titlesize = 14
legendsize = 10
ticksize = 11
LINEWIDTH = 2.2
MARKERSIZE = 7

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "tmp_latex_dir" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE = ROOT / "results/final_sweep_v7_datasweepv3/final_data_sweep_v3/arxiv_train"
SAVE_BASE = ROOT / "saves/final_sweep_v7_datasweepv3/final_data_sweep_v3/arxiv_train"

# Each row: (label, balanced-cell, natrate-cell). Each cell: (save_dir, results_dir, ckpts).
def cell(rel_save, rel_res, ckpts):
    return (SAVE_BASE / rel_save, BASE / rel_res, ckpts)

ROWS = [
    ("Text · Small (21k)",
        cell("small/arxiv_21k_text",                   "small/arxiv_21k_text",
             [656, 1312, 1968, 2624]),
        cell("natrate_sachin/arxiv_natrate_21k_text",  "natrate_sachin/arxiv_natrate_21k_text",
             [656, 1312, 1968, 2624])),
    ("Vision · Small (21k)",
        cell("small/arxiv_21k_vision",                 "small/arxiv_21k_vision",
             [1309, 2618, 3927, 5236]),
        cell("natrate_sachin/arxiv_natrate_21k_vision","natrate_sachin/arxiv_natrate_21k_vision",
             [1309, 2618, 3927, 5236])),
    ("Text · Large (71k bal / 47k natrate)",
        cell("large/arxiv_balanced_per_venue_text",    "large/arxiv_balanced_per_venue_text",
             []),  # Harish job pending
        cell("large_natrate/arxiv_natrate_balanced_per_venue_text",
             "large_natrate/arxiv_natrate_balanced_per_venue_text",
             [])), # sk7524 job pending
    ("Vision · Large (71k bal / 47k natrate)",
        cell("large/arxiv_balanced_per_venue_vision",  "large/arxiv_balanced_per_venue_vision",
             [4431, 8862, 13293]),
        cell("large_natrate/arxiv_natrate_balanced_per_venue_vision",
             "large_natrate/arxiv_natrate_balanced_per_venue_vision",
             [])), # sk7524 job pending
]

# Test-eval columns: (col_label, test_tag, val_tag_for_calibration_or_None)
COLS = [
    ("arxiv balanced (y24+)", "arxiv_balanced_test",  None),
    ("arxiv natrate (y24+)",  "arxiv_natrate_test",   "arxiv_natrate_val"),
    ("ICLR balanced (25/26)", "iclr_balanced_test",   None),
    ("ICLR natrate (25/26)",  "iclr_natrate_test",    "iclr_natrate_val"),
]

COLOR_BALANCED = "#08519C"  # blue
COLOR_NATRATE  = "#A63603"  # orange-brown


def find_jsonl(results_dir, tag, step):
    base = results_dir / tag
    for fname in (f"finetuned-ckpt-{step}.jsonl",
                  f"finetuned-ckpt-{step}-gpu-test.jsonl"):
        f = base / fname
        if f.exists() and f.stat().st_size > 0:
            return f
    return None


def label_is_accept(label_str):
    t = (label_str or "").lower()
    if "boxed{accept}" in t: return True
    if "boxed{reject}" in t: return False
    if "accept" in t: return True
    if "reject" in t: return False
    return None


def parse_record(rec):
    """Return (p_accept, gold_accept) or (None, None) if unparsable.

    Decision token = position with max |la-lr| among positions where both
    logprob_accept and logprob_reject are non-None.
    """
    la = rec.get("logprob_accept")
    lr = rec.get("logprob_reject")
    g = label_is_accept(rec.get("label", ""))
    if g is None or la is None or lr is None:
        return (None, None)
    best_k, best_d = -1, -1.0
    for k in range(min(len(la), len(lr))):
        a, r = la[k], lr[k]
        if a is None or r is None:
            continue
        # treat sentinel -1e9 as "missing"
        if a < -1e8 or r < -1e8:
            continue
        d = abs(a - r)
        if d > best_d:
            best_d = d; best_k = k
    if best_k < 0:
        return (None, g)
    a, r = la[best_k], lr[best_k]
    # 2-class softmax
    m = max(a, r)
    p_acc = math.exp(a - m) / (math.exp(a - m) + math.exp(r - m))
    return (p_acc, g)


def load_scores(jsonl_path):
    """Return list of (p_accept, gold_accept) tuples; skips unparsable rows."""
    if jsonl_path is None or not jsonl_path.exists():
        return []
    out = []
    with jsonl_path.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            p, g = parse_record(rec)
            if p is None or g is None:
                continue
            out.append((p, g))
    return out


def acc_at_threshold(scored, thr):
    if not scored:
        return None, None, None
    tp = tn = fp = fn = 0
    for p, g in scored:
        pred = (p > thr)
        if pred and g: tp += 1
        elif (not pred) and (not g): tn += 1
        elif pred and (not g): fp += 1
        elif (not pred) and g: fn += 1
    n = tp + tn + fp + fn
    if n == 0: return None, None, None
    acc  = (tp + tn) / n * 100
    accr = tp / (tp + fn) * 100 if (tp + fn) else 0.0
    rejr = tn / (tn + fp) * 100 if (tn + fp) else 0.0
    return acc, accr, rejr


def best_threshold_balanced_acc(scored):
    """Return T* maximizing balanced acc on `scored` (val split)."""
    if not scored:
        return 0.5
    # Candidate thresholds = unique P(accept) values, plus 0.0 and 1.0
    cands = sorted({p for p, _ in scored})
    cands = [0.0] + cands + [1.0]
    best_T, best_bal = 0.5, -1.0
    for T in cands:
        _, accr, rejr = acc_at_threshold(scored, T)
        if accr is None or rejr is None:
            continue
        bal = (accr + rejr) / 2
        if bal > best_bal:
            best_bal = bal; best_T = T
    return best_T


def load_train_loss(save_dir):
    log = save_dir / "trainer_log.jsonl"
    steps, epochs = [], []
    if not log.exists():
        return np.array([]), np.array([])
    with log.open() as f:
        for line in f:
            try: r = json.loads(line)
            except Exception: continue
            if "loss" not in r: continue
            steps.append(r["current_steps"])
            epochs.append(r.get("epoch", 0.0))
    return np.array(steps), np.array(epochs)


def epoch_for_step(epochs, steps, target):
    if len(steps) == 0: return float("nan")
    idx = int(np.argmin(np.abs(steps - target)))
    return float(epochs[idx])


def style(ax, title, ylabel, last_col_in_row=False):
    ax.set_xlabel("Epoch", fontsize=labelsize - 4)
    ax.set_ylabel(ylabel, fontsize=labelsize - 4)
    ax.set_title(title, fontsize=titlesize - 1, pad=6)
    ax.tick_params(axis="both", labelsize=ticksize - 1)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(1.0)


def plot_cell(ax, row, col_idx):
    row_label, bal_cell, nat_cell = row
    col_label, test_tag, val_tag = COLS[col_idx]

    plotted_any = False
    all_y = []

    for kind, (sd, rd, ckpts), color, marker in [
        ("balanced", bal_cell, COLOR_BALANCED, "o"),
        ("natrate",  nat_cell, COLOR_NATRATE,  "s"),
    ]:
        if not ckpts:
            continue
        steps, epochs = load_train_loss(sd)

        xs, raw_y, cal_y = [], [], []
        for c in ckpts:
            test_jp = find_jsonl(rd, test_tag, c)
            test_scored = load_scores(test_jp)
            if not test_scored:
                continue
            ep = epoch_for_step(epochs, steps, c) if len(steps) else float(c)
            xs.append(ep)
            raw_acc, _, _ = acc_at_threshold(test_scored, 0.5)
            raw_y.append(raw_acc if raw_acc is not None else np.nan)

            if val_tag:
                val_jp = find_jsonl(rd, val_tag, c)
                val_scored = load_scores(val_jp)
                if val_scored:
                    Tstar = best_threshold_balanced_acc(val_scored)
                    cal_acc, _, _ = acc_at_threshold(test_scored, Tstar)
                    cal_y.append(cal_acc if cal_acc is not None else np.nan)
                else:
                    cal_y.append(np.nan)
            else:
                cal_y.append(np.nan)

        if not xs:
            continue
        # Raw line: solid for balanced, dashed for natrate
        ls_raw = "-" if kind == "balanced" else "--"
        ax.plot(xs, raw_y, ls_raw, color=color, linewidth=LINEWIDTH,
                marker=marker, markersize=MARKERSIZE,
                label=f"{kind} (raw)")
        for x, y in zip(xs, raw_y):
            if not np.isnan(y):
                ax.annotate(f"{y:.1f}", (x, y), xytext=(0, 7),
                            textcoords="offset points", ha="center",
                            fontsize=ticksize - 4, color=color, fontweight="bold")
        plotted_any = True
        all_y.extend([y for y in raw_y if not np.isnan(y)])

        # Calibrated overlay only for natrate eval cols (val_tag is set)
        if val_tag and any(not np.isnan(y) for y in cal_y):
            ls_cal = ":" if kind == "balanced" else "-."
            ax.plot(xs, cal_y, ls_cal, color=color, linewidth=LINEWIDTH,
                    marker=marker, markersize=MARKERSIZE - 1, alpha=0.85,
                    markerfacecolor="white",
                    label=f"{kind} (calib.)")
            for x, y in zip(xs, cal_y):
                if not np.isnan(y):
                    ax.annotate(f"{y:.1f}", (x, y), xytext=(0, -13),
                                textcoords="offset points", ha="center",
                                fontsize=ticksize - 4, color=color,
                                fontweight="bold", alpha=0.85)
            all_y.extend([y for y in cal_y if not np.isnan(y)])

    if plotted_any:
        if all_y:
            lo, hi = min(all_y), max(all_y); span = max(hi - lo, 4)
            ax.set_ylim(max(0, lo - 0.20 * span), min(100, hi + 0.45 * span))
        ax.legend(fontsize=legendsize - 2, loc="best", framealpha=0.85,
                  ncol=1, handlelength=2.4)
    else:
        ax.text(0.5, 0.5, "no data\n(training pending)", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray", style="italic")
    style(ax, col_label, "Acc (%)")


def main():
    nrows, ncols = len(ROWS), len(COLS)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.0 * ncols, 4.2 * nrows))
    for r, row in enumerate(ROWS):
        for c in range(ncols):
            plot_cell(axes[r, c], row, c)
        # Row label as bold ylabel-prefix on col-0
        existing_y = axes[r, 0].get_ylabel()
        axes[r, 0].set_ylabel(f"{row[0]}\n\n{existing_y}",
                              fontsize=labelsize - 4, fontweight="bold")

    fig.suptitle("7B Arxiv-Trained — Balanced vs Natural-Prior Training, "
                 "4 Eval Splits  (calibrated dotted lines on natrate-eval cols)",
                 fontsize=titlesize + 4, fontweight="bold", y=1.005)
    plt.tight_layout()

    out = OUT_DIR / "7b_balanced_vs_natrate_4x4"
    plt.savefig(f"{out}.pdf", dpi=200, bbox_inches="tight", pad_inches=0.2)
    plt.savefig(f"{out}.png", dpi=140, bbox_inches="tight", pad_inches=0.2)
    plt.close()
    print(f"Saved: {out}.pdf / .png")

    # Coverage + numeric summary
    print("\n=== Coverage & per-ckpt numbers ===")
    print(f"{'row':<42} {'train':<9} {'ckpt':>6} {'ep':>5} "
          f"{'arxBal':>7} {'arxNat':>7} {'arxNat-c':>9} "
          f"{'iclrBal':>8} {'iclrNat':>8} {'iclrNat-c':>10}")
    for row_label, bal_cell, nat_cell in ROWS:
        for kind, (sd, rd, ckpts) in [("balanced", bal_cell), ("natrate", nat_cell)]:
            if not ckpts:
                print(f"{row_label:<42} {kind:<9}   (no data)")
                continue
            steps, epochs = load_train_loss(sd)
            for c in ckpts:
                ep = epoch_for_step(epochs, steps, c) if len(steps) else float("nan")
                row_vals = [c, ep]
                for col_idx, (_, test_tag, val_tag) in enumerate(COLS):
                    test_scored = load_scores(find_jsonl(rd, test_tag, c))
                    raw_acc = acc_at_threshold(test_scored, 0.5)[0] if test_scored else None
                    row_vals.append(raw_acc)
                    if val_tag:
                        val_scored = load_scores(find_jsonl(rd, val_tag, c))
                        if val_scored and test_scored:
                            T = best_threshold_balanced_acc(val_scored)
                            cal_acc = acc_at_threshold(test_scored, T)[0]
                            row_vals.append(cal_acc)
                        else:
                            row_vals.append(None)
                def fmt(v, w):
                    return ("--" if v is None else f"{v:.1f}").rjust(w)
                ckpt, ep_ = row_vals[0], row_vals[1]
                arxBal, arxNat, arxNatC, iclrBal, iclrNat, iclrNatC = row_vals[2:]
                print(f"{row_label:<42} {kind:<9} {ckpt:>6} {ep_:>5.2f} "
                      f"{fmt(arxBal,7)} {fmt(arxNat,7)} {fmt(arxNatC,9)} "
                      f"{fmt(iclrBal,8)} {fmt(iclrNat,8)} {fmt(iclrNatC,10)}")


if __name__ == "__main__":
    main()
