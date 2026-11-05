#!/usr/bin/env python3
"""
Generate 1x3 calibration analysis figure for the LaTeX paper.

Panels:
  (a) Reliability + Coverage — merged: left y-axis = accuracy (reliability bins),
      right y-axis = coverage (dashed lines)
  (b) 3D surface: Confidence × pct_rating → accuracy (vision)
      with peak-region markers at high/low rating extremes
  (c) 3D surface: Confidence × pct_citation → accuracy (vision)
      with peak-region markers at high/low citation extremes

Output:
    tmp_latex_dir/figures/calibration.pdf / .png
"""

import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm  # noqa: F401
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
from scipy.ndimage import gaussian_filter

# ---------------------------------------------------------------------------
# Style guide
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "DejaVu Sans"],
})

labelsize = 20
titlesize = 20
legendsize = 20
ticksize = 16

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "tmp_latex_dir" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VISION_PRED = (
    ROOT / "results/final_sweep_v7_datasweepv3/optim_search_2026"
    / "bz16_lr1e-6_vision/finetuned-ckpt-2648.jsonl"
)
TEXT_PRED = (
    ROOT / "results/final_sweep_v7_datasweepv3/optim_search_2026"
    / "bz32_lr1e-6_text/finetuned-ckpt-1322.jsonl"
)
VISION_META = (
    ROOT / "data/iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_labelfix_v7_filtered_filtered24480_test/data.json"
)

DECISION_TOKEN_IDX = 5

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
BLUE = "#6098FF"
ORANGE = "#FECC81"

LINEWIDTH = 3.0
MARKERSIZE = 10

CMAP = plt.cm.RdYlGn
CNORM = mpl.colors.Normalize(vmin=0.4, vmax=1.0)

TITLE_Y = 0.97

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
    confidences, correct, preds = [], [], []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            pred = extract_prediction(rec["predict"])
            label = extract_prediction(rec["label"])
            conf = math.exp(rec["token_logprobs"][DECISION_TOKEN_IDX])
            confidences.append(conf)
            correct.append(pred == label and pred != "unknown")
            preds.append(pred)
    return np.array(confidences), np.array(correct), preds


def load_metadata(path: Path):
    with open(path) as f:
        data = json.load(f)
    return [d["_metadata"] for d in data]


# ---------------------------------------------------------------------------
# Panel (a): Reliability + Coverage (merged, dual y-axis)
# ---------------------------------------------------------------------------

def panel_reliability_coverage(ax, text_conf, text_corr, vis_conf, vis_corr):
    """Reliability bins (solid+markers) on left y, coverage (dashed) on right y."""

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

    # Perfect calibration
    ax.plot([0.5, 1.0], [0.5, 1.0], "-", color="grey", linewidth=1.5)
    ax.set_xlabel("Confidence", fontsize=labelsize)
    ax.set_ylabel("Accuracy", fontsize=labelsize)
    ax.set_xlim(0.48, 1.02)
    ax.set_ylim(0.45, 1.05)
    ax.tick_params(axis="both", labelsize=ticksize)
    ax.grid(True, linestyle="--")

    # Coverage curve (right y-axis) — DASHED
    ax2 = ax.twinx()
    thresholds = np.linspace(0.5, 0.99, 200)

    for conf, color in [
        (text_conf, BLUE),
        (vis_conf, ORANGE),
    ]:
        coverages = [(conf >= t).mean() for t in thresholds]
        ax2.plot(thresholds, coverages, color=color, linestyle="--",
                 linewidth=LINEWIDTH - 0.5, alpha=0.6)

    ax2.set_ylabel("Coverage", fontsize=labelsize, color="#666666")
    ax2.set_ylim(0, 1.05)
    ax2.tick_params(axis="y", labelsize=ticksize, colors="#666666")

    # Vertical reference lines at t=0.7, 0.8, 0.9
    for t_val in [0.7, 0.8, 0.9]:
        ax.axvline(x=t_val, color="#BBBBBB", linewidth=1.0, linestyle=":",
                   alpha=0.6, zorder=0)


# ---------------------------------------------------------------------------
# Panels (b)/(c): 3D surface plots with peak markers
# ---------------------------------------------------------------------------

def find_peak_regions(x_arr, conf_arr, corr_arr, x_bins, y_bins):
    """Find the peak-accuracy point on the *smoothed* surface in low-x and high-x regions."""
    x_edges = np.linspace(0, 1, x_bins + 1)
    y_edges = np.linspace(0.5, 1.0, y_bins + 1)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    acc_grid = np.full((y_bins, x_bins), np.nan)
    for xi in range(x_bins):
        for yi in range(y_bins):
            x_lo, x_hi = x_edges[xi], x_edges[xi + 1]
            y_lo, y_hi = y_edges[yi], y_edges[yi + 1]
            x_mask = (x_arr >= x_lo) & ((x_arr < x_hi) if xi < x_bins - 1 else (x_arr <= x_hi))
            y_mask = (conf_arr >= y_lo) & ((conf_arr < y_hi) if yi < y_bins - 1 else (conf_arr <= y_hi))
            mask = x_mask & y_mask
            if mask.sum() >= 3:
                acc_grid[yi, xi] = corr_arr[mask].mean()

    # Use the smoothed surface so peaks reflect real trends, not noisy bins
    global_mean = np.nanmean(acc_grid) if np.any(~np.isnan(acc_grid)) else 0.5
    acc_filled = np.where(np.isnan(acc_grid), global_mean, acc_grid)
    acc_smooth = gaussian_filter(acc_filled, sigma=1.0)

    peaks = []
    for region_name, x_filter in [("low", lambda xc: xc < 0.3),
                                   ("high", lambda xc: xc > 0.7)]:
        best_acc, best_x, best_y = -1, 0, 0
        for xi in range(x_bins):
            if not x_filter(x_centers[xi]):
                continue
            for yi in range(y_bins):
                if acc_smooth[yi, xi] > best_acc:
                    best_acc = acc_smooth[yi, xi]
                    best_x = x_centers[xi]
                    best_y = y_centers[yi]
        if best_acc > 0:
            peaks.append((region_name, best_x, best_y, best_acc))

    return peaks


def panel_surface(fig, subplot_pos, conf, corr, meta_values,
                  xlabel, x_bins=15, y_bins=15):
    """3D surface: x=meta_value, y=confidence, z=accuracy, with peak markers."""
    ax = fig.add_subplot(subplot_pos, projection="3d")

    x = np.array(meta_values, dtype=float)
    y = conf.copy()

    x_edges = np.linspace(0, 1, x_bins + 1)
    y_edges = np.linspace(0.5, 1.0, y_bins + 1)

    acc_grid = np.full((y_bins, x_bins), np.nan)

    for xi in range(x_bins):
        for yi in range(y_bins):
            x_lo, x_hi = x_edges[xi], x_edges[xi + 1]
            y_lo, y_hi = y_edges[yi], y_edges[yi + 1]
            x_mask = (x >= x_lo) & ((x < x_hi) if xi < x_bins - 1 else (x <= x_hi))
            y_mask = (y >= y_lo) & ((y < y_hi) if yi < y_bins - 1 else (y <= y_hi))
            mask = x_mask & y_mask
            if mask.sum() >= 3:
                acc_grid[yi, xi] = corr[mask].mean()

    global_mean = np.nanmean(acc_grid) if np.any(~np.isnan(acc_grid)) else 0.5
    acc_filled = np.where(np.isnan(acc_grid), global_mean, acc_grid)
    acc_smooth = gaussian_filter(acc_filled, sigma=1.0)

    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    X, Y = np.meshgrid(x_centers, y_centers)

    ax.plot_surface(
        X, Y, acc_smooth,
        facecolors=CMAP(CNORM(acc_smooth)),
        edgecolor="grey", linewidth=0.3, alpha=0.85,
        shade=True, antialiased=True,
    )

    ax.contourf(
        X, Y, acc_smooth,
        levels=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        cmap=CMAP, alpha=0.3, offset=0.4,
    )

    # Peak markers: vertical dashed line + X at top + accuracy label
    peaks = find_peak_regions(x, y, corr, x_bins, y_bins)
    z_top = 1.0
    for region_name, px, py, pacc in peaks:
        # Vertical dashed line from floor to top
        ax.plot([px, px], [py, py], [0.4, z_top],
                linestyle="--", color="black", linewidth=1.5, alpha=0.7,
                zorder=10)
        # X marker at the top
        ax.scatter([px], [py], [z_top], marker="x", color="black",
                   s=120, linewidths=2.5, zorder=11)
        # Accuracy label
        ax.text(px, py, z_top + 0.02, f"{pacc:.0%}",
                ha="center", va="bottom", fontsize=14, fontweight="bold",
                color="black", zorder=12)
        # Horizontal dashed line to help read accuracy off z-axis
        ax.plot([px, px], [py, 0.5], [pacc, pacc],
                linestyle=":", color="black", linewidth=1.0, alpha=0.5,
                zorder=9)

    ax.set_xlabel(xlabel, fontsize=labelsize - 4, labelpad=10)
    ax.set_ylabel("Confidence", fontsize=labelsize - 4, labelpad=10)
    ax.set_zlabel("Accuracy", fontsize=labelsize - 4, labelpad=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0.5, 1.0)
    ax.set_zlim(0.4, 1.0)
    ax.tick_params(labelsize=ticksize - 4)
    ax.view_init(elev=25, azim=-50)

    return ax


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    text_conf, text_corr, _ = load_predictions(TEXT_PRED)
    vis_conf, vis_corr, _ = load_predictions(VISION_PRED)
    vis_meta = load_metadata(VISION_META)
    assert len(vis_conf) == len(vis_meta)

    vis_pct_rating = [m["pct_rating"] for m in vis_meta]
    vis_pct_citation = [m["citation_normalized_by_year"] for m in vis_meta]

    # --- Layout -----------------------------------------------------------
    fig = plt.figure(figsize=(24, 7.5))

    gs = fig.add_gridspec(
        2, 3,
        width_ratios=[1.3, 1, 1],
        height_ratios=[0.06, 1],
        hspace=0.08, wspace=0.30,
    )

    # -- Top row: legend for (a), shared colorbar for (b)+(c) ---------------

    # Legend axis above panel (a)
    ax_leg = fig.add_subplot(gs[0, 0])
    ax_leg.set_axis_off()
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
    ax_leg.legend(handles=legend_handles, loc="center", ncol=4,
                  fontsize=legendsize - 4, frameon=False, handlelength=2.0)

    # Colorbar axis above panels (b) and (c)
    cbar_ax = fig.add_subplot(gs[0, 1:])
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=CNORM)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cb.set_label("Accuracy", fontsize=labelsize - 2, labelpad=4)
    cb.ax.tick_params(labelsize=ticksize - 2)
    cb.ax.xaxis.set_ticks_position("top")
    cb.ax.xaxis.set_label_position("top")

    # -- Bottom row: the 3 panels ------------------------------------------

    # Panel (a)
    ax_a = fig.add_subplot(gs[1, 0])
    panel_reliability_coverage(ax_a, text_conf, text_corr, vis_conf, vis_corr)

    # Panel (b)
    ax_b = panel_surface(fig, gs[1, 1], vis_conf, vis_corr,
                         vis_pct_rating, xlabel="Pctl. Rating")

    # Panel (c)
    ax_c = panel_surface(fig, gs[1, 2], vis_conf, vis_corr,
                         vis_pct_citation, xlabel="Pctl. Citation")

    # -- Titles via fig.text so they are vertically aligned -----------------
    for col_idx, label in [(0, "(a) Reliability & Coverage"),
                           (1, "(b) Confidence × Rating"),
                           (2, "(c) Confidence × Citation")]:
        widths = np.array([1.3, 1, 1])
        cumw = np.concatenate([[0], np.cumsum(widths)])
        wspace_frac = 0.30
        panel_total = cumw[-1] + wspace_frac * (len(widths) - 1)
        left = (cumw[col_idx] + wspace_frac * col_idx) / panel_total
        right = (cumw[col_idx + 1] + wspace_frac * col_idx) / panel_total
        x_center = (left + right) / 2
        fig.text(x_center, TITLE_Y, label,
                 ha="center", va="bottom",
                 fontsize=titlesize, fontweight="bold")

    # ----------------------------------------------------------------------
    out = OUTPUT_DIR / "calibration.pdf"
    plt.savefig(out, dpi=200, bbox_inches="tight", transparent=False)
    plt.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight",
                transparent=False)
    plt.close()
    print(f"Saved: {out}")
    print(f"Saved: {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
