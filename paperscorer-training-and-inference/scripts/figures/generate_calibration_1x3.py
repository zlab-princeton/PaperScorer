#!/usr/bin/env python3
"""
1×3 calibration figure:
  (a) Reliability & Coverage (Platt-scaled) — wider
  (b) Impact Correlation: pctl_citation vs quality signals (Human rating, Text p(accept), Vision p(accept))
  (c) 3D surface: Confidence × Pctl. Rating → Accuracy (colorbar below)

Output: calibration_1x3.pdf / .png
"""

import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
from scipy import stats
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize
from scipy.special import expit as sigmoid

# ── Lab style ──
mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "helvetica",
})

labelsize = 28
titlesize = 34
legendsize = 20
ticksize = 20

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "tmp_latex_dir" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Lab palette ──
BLUE = "#6098FF"
ORANGE = "#FECC81"
GREEN = "#77B25D"
RED = "#FF8988"
PURPLE = "#B28CFF"

LINEWIDTH = 3.0
MARKERSIZE = 10
DECISION_TOKEN_IDX = 5

# ── Paths (main table models) ──
TEXT_TEST_PRED = (
    ROOT / "results/final_sweep_v7_datasweepv3/optim_search_2026"
    / "bz32_lr1e-6_text/finetuned-ckpt-1322.jsonl"
)
TEXT_VAL_PRED = (
    ROOT / "results/final_sweep_v7_datasweepv3/optim_search_2026"
    / "bz32_lr1e-6_text/validation-ckpt-1322.jsonl"
)
VISION_TEST_PRED = (
    ROOT / "results/final_sweep_v7_datasweepv3/optim_search_2026"
    / "bz16_lr1e-6_vision/finetuned-ckpt-2648.jsonl"
)
VISION_VAL_PRED = (
    ROOT / "results/final_sweep_v7_datasweepv3/optim_search_2026"
    / "bz16_lr1e-6_vision/validation-ckpt-2648.jsonl"
)
TEXT_META = (
    ROOT / "data/iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_test/data.json"
)
VISION_META = (
    ROOT / "data/iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_labelfix_v7_filtered_filtered24480_test/data.json"
)

CMAP = plt.cm.RdYlGn
CNORM = mpl.colors.Normalize(vmin=0.4, vmax=1.0)
DENSITY_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "density_focus",
    ["#fff8dc", "#d7e8a6", "#78b66a", "#216b45"],
)
DENSITY_GAMMA = 0.7
REGION_SPECS = [
    {
        "label": "A",
        "x_bounds": (0.0, 0.4),
        "y_bounds": (0.8, 1.0),
        "color": BLUE,
    },
    {
        "label": "B",
        "x_bounds": (0.8, 1.0),
        "y_bounds": (0.65, 0.9),
        "color": RED,
    },
]


# =========================================================================
# Helpers
# =========================================================================

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
    """Return (confidences, correct, logits)."""
    confidences, correct, logits = [], [], []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            pred = extract_prediction(rec["predict"])
            label = extract_prediction(rec["label"])
            conf = math.exp(rec["token_logprobs"][DECISION_TOKEN_IDX])
            conf = np.clip(conf, 1e-7, 1 - 1e-7)
            logit = math.log(conf / (1 - conf))
            confidences.append(conf)
            logits.append(logit)
            correct.append(pred == label and pred != "unknown")
    return np.array(confidences), np.array(correct), np.array(logits)


def load_full_data(pred_path: Path, meta_path: Path):
    """Load p(accept), pct_rating, pct_citation, year."""
    preds = [json.loads(l) for l in open(pred_path)]
    test_data = json.load(open(meta_path))
    n = min(len(preds), len(test_data))

    p_accept, pct_rating, pct_citation, years = [], [], [], []
    for i in range(n):
        rec = preds[i]
        m = test_data[i].get("_metadata", {})
        pr = m.get("pct_rating")
        pc = m.get("citation_normalized_by_year")
        year = m.get("year")
        if pr is None or pc is None or year is None:
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
        pct_rating.append(pr)
        pct_citation.append(pc)
        years.append(year)

    return (np.array(p_accept), np.array(pct_rating),
            np.array(pct_citation), np.array(years))


def load_metadata(path: Path):
    with open(path) as f:
        data = json.load(f)
    return [d["_metadata"] for d in data]


def fit_platt(val_logits, val_correct):
    def nll(params):
        a, b = params
        p = sigmoid(a * val_logits + b)
        pc = np.where(val_correct, p, 1 - p)
        return -np.log(np.clip(pc, 1e-10, 1.0)).mean()
    result = minimize(nll, x0=[1.0, 0.0], method="Nelder-Mead")
    return result.x


def apply_platt(logits, a, b):
    return sigmoid(np.abs(a * logits + b))


# =========================================================================
# Panel (a): Reliability & Coverage
# =========================================================================

def panel_reliability_coverage(ax, text_conf, text_corr, vis_conf, vis_corr):
    bins = np.linspace(0.5, 1.0, 11)
    for conf, corr, color, label in [
        (text_conf, text_corr, BLUE, "Text"),
        (vis_conf, vis_corr, ORANGE, "Vision"),
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
                linewidth=LINEWIDTH, markersize=MARKERSIZE, alpha=0.9,
                label=label)

    ax.plot([0.5, 1.0], [0.5, 1.0], "-", color="grey", linewidth=1.5)
    ax.set_xlabel("Binned Platt Confidence", fontsize=labelsize)
    ax.set_ylabel("Accuracy", fontsize=labelsize, labelpad=12)
    ax.set_xlim(0.48, 1.02)
    ax.set_ylim(0.45, 1.05)
    ax.tick_params(axis="both", labelsize=ticksize)
    ax.grid(False)

    ax2 = ax.twinx()
    thresholds = np.linspace(0.5, 0.99, 200)
    for conf, color in [(text_conf, BLUE), (vis_conf, ORANGE)]:
        coverages = [(conf >= t).mean() for t in thresholds]
        ax2.plot(thresholds, coverages, color=color, linestyle="--",
                 linewidth=LINEWIDTH - 0.5, alpha=0.6)
    ax2.set_ylabel("Cumulative Coverage", fontsize=labelsize, color="#666666", labelpad=12)
    ax2.set_ylim(0, 1.05)
    ax2.tick_params(axis="y", labelsize=ticksize, colors="#666666")

    for t_val in [0.7, 0.8, 0.9]:
        ax.axvline(x=t_val, color="#BBBBBB", linewidth=1.0, linestyle=":",
                   alpha=0.6, zorder=0)

    coverage_handle = Line2D([0], [0], color="#888888", linewidth=LINEWIDTH - 0.5,
                             linestyle="--", label="Coverage")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(coverage_handle)
    labels.append("Coverage")
    ax.legend(handles, labels, fontsize=legendsize, loc="upper center",
              framealpha=0.95, ncol=3)

    vis_thresh = 0.7
    vis_mask = vis_conf >= vis_thresh
    vis_acc = vis_corr[vis_mask].mean()
    guide_cov = 0.30
    guide_color = "#444444"
    ax2.plot([vis_thresh, vis_thresh], [0.0, guide_cov],
             linestyle=":", color=guide_color, linewidth=2.2, zorder=6)
    ax2.plot([vis_thresh, 0.99], [guide_cov, guide_cov],
             linestyle=":", color=guide_color, linewidth=2.2, zorder=6)
    ax2.text(
        0.84, guide_cov + 0.01,
        f"{100 * vis_acc:.0f}\\% acc. on 30\\% of data @ conf $\\geq$ 0.7",
        fontsize=ticksize,
        color="black",
        ha="center",
        va="bottom",
        zorder=7,
    )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)


# =========================================================================
# Panel (b): Impact Correlation — quality signal vs pctl_citation
# =========================================================================

def _binned_means(x, y, n_bins=10):
    """Return bin centers and means for plotting."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    centers, means = [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (x >= lo) & (x < hi) if hi < 1.0 else (x >= lo) & (x <= hi)
        if mask.sum() >= 5:
            centers.append(x[mask].mean())
            means.append(y[mask].mean())
    return centers, means


def panel_impact_correlation(ax, text_pa, text_pr, text_pc,
                             vis_pa, vis_pr, vis_pc):
    """Linear fits + binned means: quality signal (x) vs pctl_citation (y)."""
    x_fit = np.linspace(0, 1, 100)

    # Each line: bin papers by that quality signal, plot mean citation per bin
    # Same underlying papers, different x-axis assignments
    for x_vals, color, ls, label_name, r_label in [
        (vis_pr, GREEN, "-", r"Human \textit{Pctl.\ Rating}", None),
        (text_pa, BLUE, "--", "PaperLens-Text $p(\\mathrm{accept})$", None),
        (vis_pa, ORANGE, "--", "PaperLens-Vision $p(\\mathrm{accept})$", None),
    ]:
        y_vals = vis_pc if x_vals is vis_pr or x_vals is vis_pa else text_pc
        r, _ = stats.pearsonr(x_vals, y_vals)
        slope, intercept = np.polyfit(x_vals, y_vals, 1)
        cx, cy = _binned_means(x_vals, y_vals)
        ax.plot(cx, cy, "o", color="black", markersize=8, alpha=0.5,
                zorder=3)
        ax.plot(x_fit, slope * x_fit + intercept, color=color, linewidth=LINEWIDTH,
                linestyle=ls, label=f"{label_name} ($r$={r:.2f})", zorder=5)
        print(f"  {label_name}: r={r:.3f}")

    # Add legend entry for the binned empirical means
    from matplotlib.lines import Line2D
    dot_handle = Line2D([0], [0], marker="o", color="w", markerfacecolor="black",
                        markersize=8, alpha=0.5, label="Binned Ground-Truth Mean")
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0, dot_handle)
    labels.insert(0, "Binned Ground-Truth Mean")

    ax.set_xlabel("Quality Signal", fontsize=labelsize)
    ax.set_ylabel("Pctl.\\ Citation", fontsize=labelsize, labelpad=12)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.2, 0.75)
    ax.tick_params(axis="both", labelsize=ticksize)
    ax.grid(False)
    ax.legend(handles, labels, fontsize=legendsize - 2, loc="lower right", framealpha=0.95)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)


# =========================================================================
# Panel (c): 3D surface — Confidence × Rating → Accuracy
# =========================================================================

def panel_surface(fig, subplot_pos, conf, corr, meta_values,
                  x_bins=15, y_bins=15, min_count=3):
    ax = fig.add_subplot(subplot_pos, projection="3d")

    x = np.array(meta_values, dtype=float)
    y = conf.copy()

    x_edges = np.linspace(0, 1, x_bins + 1)
    y_edges = np.linspace(0.5, 1.0, y_bins + 1)

    acc_grid = np.full((y_bins, x_bins), np.nan)
    count_grid = np.zeros((y_bins, x_bins), dtype=int)
    for xi in range(x_bins):
        for yi in range(y_bins):
            x_lo, x_hi = x_edges[xi], x_edges[xi + 1]
            y_lo, y_hi = y_edges[yi], y_edges[yi + 1]
            x_mask = (x >= x_lo) & ((x < x_hi) if xi < x_bins - 1 else (x <= x_hi))
            y_mask = (y >= y_lo) & ((y < y_hi) if yi < y_bins - 1 else (y <= y_hi))
            mask = x_mask & y_mask
            count_grid[yi, xi] = mask.sum()
            if mask.sum() >= min_count:
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

    ax.set_xlabel("Pctl.\\ Rating", fontsize=labelsize - 2, labelpad=18)
    ax.set_ylabel("Confidence", fontsize=labelsize - 2, labelpad=18)
    ax.set_zlabel("")  # hide default, place manually
    ax.set_xlim(0, 1)
    ax.set_ylim(0.5, 1.0)
    ax.set_zlim(0.4, 1.0)
    ax.set_yticks([0.6, 0.7, 0.8, 0.9])
    ax.tick_params(labelsize=ticksize - 2, pad=4)
    ax.view_init(elev=30, azim=-50)
    # Manually place z-label as text annotation — pushed right of tick labels
    ax.text2D(1.08, 0.55, "Accuracy", transform=ax.transAxes,
              fontsize=labelsize - 2, rotation=90, ha="left", va="center")

    # Peak within the high-confidence / high-rating region.
    best_peak = None
    for xi, x_val in enumerate(x_centers):
        if x_val < 0.75:
            continue
        for yi, y_val in enumerate(y_centers):
            if y_val < 0.8:
                continue
            cand = (float(acc_smooth[yi, xi]), float(x_val), float(y_val))
            if best_peak is None or cand[0] > best_peak[0]:
                best_peak = cand
    if best_peak is not None:
        peak_acc, peak_x, peak_y = best_peak
        ax.plot(
            [peak_x, peak_x], [peak_y, peak_y], [0.4, 1.0],
            linestyle="--", color="black", linewidth=2.0, alpha=0.85, zorder=18,
        )
        ax.scatter(
            [peak_x], [peak_y], [1.0],
            marker="x", color="black", s=120, linewidths=2.5,
            depthshade=False, zorder=19,
        )
        ax.text(
            peak_x, peak_y, 1.02,
            rf"\textbf{{{100 * peak_acc:.0f}}}",
            color="black", fontsize=labelsize - 4, fontweight="bold",
            ha="center", va="bottom", zorder=20,
        )

    return ax, {
        "x_centers": x_centers,
        "y_centers": y_centers,
        "acc_smooth": acc_smooth,
        "z_top": 1.0,
    }


# =========================================================================
# Panel (d): 3D density bars — sample count per (rating, confidence) bin
# =========================================================================


def compute_density_grid(x, y, x_bins=8, y_bins=8):
    x_edges = np.linspace(0, 1, x_bins + 1)
    y_edges = np.linspace(0.5, 1.0, y_bins + 1)

    count_grid = np.zeros((y_bins, x_bins), dtype=int)
    for xi in range(x_bins):
        for yi in range(y_bins):
            x_lo, x_hi = x_edges[xi], x_edges[xi + 1]
            y_lo, y_hi = y_edges[yi], y_edges[yi + 1]
            x_mask = (x >= x_lo) & ((x < x_hi) if xi < x_bins - 1 else (x <= x_hi))
            y_mask = (y >= y_lo) & ((y < y_hi) if yi < y_bins - 1 else (y <= y_hi))
            count_grid[yi, xi] = (x_mask & y_mask).sum()

    return count_grid, x_edges, y_edges


def panel_density_bars(fig, subplot_pos, conf, meta_values,
                       x_bins=8, y_bins=8):
    ax = fig.add_subplot(subplot_pos, projection="3d")

    x = np.array(meta_values, dtype=float)
    y = conf.copy()

    count_grid, x_edges, y_edges = compute_density_grid(x, y, x_bins=x_bins, y_bins=y_bins)

    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    dx = x_edges[1] - x_edges[0]
    dy = y_edges[1] - y_edges[0]

    max_count = count_grid.max()
    density_norm = mpl.colors.PowerNorm(gamma=DENSITY_GAMMA, vmin=0, vmax=max_count)

    # Same orientation as panel (c): elev=30, azim=-50
    # Draw back-to-front for this angle
    azim_rad = np.deg2rad(-50)
    bars = []
    for xi in range(x_bins):
        for yi in range(y_bins):
            cnt = count_grid[yi, xi]
            if cnt == 0:
                continue
            dist = x_centers[xi] * np.cos(azim_rad) + y_centers[yi] * np.sin(azim_rad)
            bars.append((dist, xi, yi, cnt))
    bars.sort(key=lambda b: b[0])

    inset_x = dx * 0.05
    inset_y = dy * 0.05
    for _, xi, yi, cnt in bars:
        color = DENSITY_CMAP(density_norm(cnt))
        ax.bar3d(
            x_edges[xi] + inset_x, y_edges[yi] + inset_y, 0,
            dx - 2 * inset_x, dy - 2 * inset_y, cnt,
            color=color, edgecolor="#5b6b5b", linewidth=0.3, alpha=0.95,
            shade=False,
        )

    ax.set_xlabel("Pctl.\\ Rating", fontsize=labelsize - 2, labelpad=18)
    ax.set_ylabel("Confidence", fontsize=labelsize - 2, labelpad=18)
    ax.set_zlabel("")
    ax.set_xlim(0, 1)
    ax.set_ylim(0.5, 1.0)
    ax.set_zlim(0, max_count)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9])
    ax.tick_params(labelsize=ticksize - 2, pad=4)
    ax.view_init(elev=30, azim=-50)
    ax.text2D(1.08, 0.55, "Frequency", transform=ax.transAxes,
              fontsize=labelsize - 2, rotation=90, ha="left", va="center")

    return ax, {
        "count_grid": count_grid,
        "z_top": max_count,
    }


def surface_height_at_point(surface_info, x, y):
    x_idx = int(np.argmin(np.abs(surface_info["x_centers"] - x)))
    y_idx = int(np.argmin(np.abs(surface_info["y_centers"] - y)))
    return float(surface_info["acc_smooth"][y_idx, x_idx])


def draw_surface_stems(ax_surface, surface_info):
    surface_top = surface_info["z_top"]
    surface_floor = 0.4

    for spec in REGION_SPECS[:1]:
        x_bounds = spec["x_bounds"]
        y_bounds = spec["y_bounds"]
        center_x = 0.5 * (x_bounds[0] + x_bounds[1])
        center_y = 0.5 * (y_bounds[0] + y_bounds[1])
        center_z = surface_height_at_point(surface_info, center_x, center_y)
        ax_surface.plot(
            [center_x, center_x], [center_y, center_y], [surface_floor, surface_top],
            linestyle="--", color="black", linewidth=2.0, alpha=0.8, zorder=18,
        )
        ax_surface.scatter(
            [center_x], [center_y], [surface_top],
            marker="x", color="black", s=120, linewidths=2.5,
            depthshade=False, zorder=19,
        )
        ax_surface.text(
            center_x, center_y, 1.02,
            rf"\textbf{{{100 * center_z:.0f}}}",
            color="black", fontsize=labelsize - 4, fontweight="bold",
            ha="center", va="bottom", zorder=20,
        )


# =========================================================================
# Main
# =========================================================================

def main():
    # --- Platt fitting ---
    text_val_conf, text_val_corr, text_val_logits = load_predictions(TEXT_VAL_PRED)
    vis_val_conf, vis_val_corr, vis_val_logits = load_predictions(VISION_VAL_PRED)
    text_a, text_b = fit_platt(text_val_logits, text_val_corr)
    vis_a, vis_b = fit_platt(vis_val_logits, vis_val_corr)
    print(f"Text Platt:   a={text_a:.3f}, b={text_b:.3f}")
    print(f"Vision Platt: a={vis_a:.3f}, b={vis_b:.3f}")

    # --- Test predictions ---
    text_conf_raw, text_corr, text_logits = load_predictions(TEXT_TEST_PRED)
    vis_conf_raw, vis_corr, vis_logits = load_predictions(VISION_TEST_PRED)
    vis_meta = load_metadata(VISION_META)

    text_conf_platt = apply_platt(text_logits, text_a, text_b)
    vis_conf_platt = apply_platt(vis_logits, vis_a, vis_b)

    # --- Threshold table ---
    print("\n=== Threshold Table (Platt-scaled) ===")
    for t in [0.5, 0.6, 0.7, 0.8, 0.9]:
        for name, conf, corr in [("Text", text_conf_platt, text_corr),
                                   ("Vision", vis_conf_platt, vis_corr)]:
            mask = conf >= t
            acc = corr[mask].mean() if mask.sum() > 0 else 0
            cov = mask.mean()
            print(f"  {name} >= {t:.1f}: acc={acc:.1%}, coverage={cov:.1%} ({mask.sum()}/{len(conf)})")

    # --- Full data for panel (b) — ICLR 2025 only ---
    print("\n=== Impact Correlation (ICLR 2025) ===")
    text_pa, text_pr, text_pc, text_yrs = load_full_data(TEXT_TEST_PRED, TEXT_META)
    vis_pa, vis_pr, vis_pc, vis_yrs = load_full_data(VISION_TEST_PRED, VISION_META)

    text_no26 = text_yrs == 2025
    vis_no26 = vis_yrs == 2025

    # --- Layout: 2 rows × 3 cols ---
    # Row 0: (a) reliability, (b) impact, (c) 3D accuracy surface
    # Row 1: empty, empty, colorbar under (c)
    fig = plt.figure(figsize=(40, 12))
    gs = fig.add_gridspec(
        2, 3,
        width_ratios=[1.9, 1.55, 1.35],
        height_ratios=[1, 0.05],
        hspace=0.18, wspace=0.34,
    )

    # Panel (a)
    ax_a = fig.add_subplot(gs[0, 0])
    panel_reliability_coverage(ax_a, text_conf_platt, text_corr,
                               vis_conf_platt, vis_corr)

    # Panel (b) — Impact Correlation (ICLR 2025)
    ax_b = fig.add_subplot(gs[0, 1])
    panel_impact_correlation(ax_b,
                             text_pa[text_no26], text_pr[text_no26], text_pc[text_no26],
                             vis_pa[vis_no26], vis_pr[vis_no26], vis_pc[vis_no26])

    # Panel (c) — 3D accuracy surface
    vis_pct_rating = [m["pct_rating"] for m in vis_meta]
    ax_c, surface_info = panel_surface(fig, gs[0, 2], vis_conf_raw, vis_corr, vis_pct_rating)
    pos_c = ax_c.get_position()
    ax_c.set_position([pos_c.x0 - 0.01, pos_c.y0 + 0.015, pos_c.width * 1.12, pos_c.height * 1.10])

    # Colorbar under panel (c) — Accuracy
    cbar_acc_ax = fig.add_subplot(gs[1, 2])
    pos_cb = cbar_acc_ax.get_position()
    cbar_acc_ax.set_position([pos_cb.x0 + 0.005, pos_cb.y0 + 0.09, pos_cb.width * 0.98, pos_cb.height])
    sm_acc = plt.cm.ScalarMappable(cmap=CMAP, norm=CNORM)
    sm_acc.set_array([])
    cb_acc = fig.colorbar(sm_acc, cax=cbar_acc_ax, orientation="horizontal")
    cb_acc.set_label("Accuracy", fontsize=labelsize - 2, labelpad=4)
    cb_acc.ax.tick_params(labelsize=ticksize - 2)

    # Hide bottom row for (a) and (b)
    for col in [0, 1]:
        ax_empty = fig.add_subplot(gs[1, col])
        ax_empty.set_axis_off()

    # --- Aligned titles at same y ---
    fig.canvas.draw()
    draw_surface_stems(ax_c, surface_info)

    title_y = 0.935
    for ax_obj, label in [
        (ax_a, "(a) Reliability \\& Coverage"),
        (ax_b, "(b) Impact Correlation (ICLR 2025)"),
        (ax_c, "(c) Accuracy Surface (Vision)"),
    ]:
        pos = ax_obj.get_position()
        x_center = (pos.x0 + pos.x1) / 2
        fig.text(x_center, title_y, label,
                 ha="center", va="bottom",
                 fontsize=titlesize, fontweight="bold")

    out = OUTPUT_DIR / "calibration_1x3"
    plt.savefig(f"{out}.pdf", dpi=200, bbox_inches="tight",
                pad_inches=0.3, transparent=False)
    plt.savefig(f"{out}.png", dpi=150, bbox_inches="tight",
                pad_inches=0.3, transparent=False)
    plt.close()
    print(f"\nSaved: {out}.pdf")
    print(f"Saved: {out}.png")


if __name__ == "__main__":
    main()
