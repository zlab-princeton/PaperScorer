#!/usr/bin/env python3
"""
Before/after calibration plots for post-hoc calibration methods.

2×2 grid: columns = Text / Vision, rows = per-bin reliability / cumulative accuracy.
Each panel overlays: uncalibrated + all 4 calibration methods.

Usage: uv run scripts/tmp_latex_dir/generate_calibration_posthoc.py
"""

import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.optimize import minimize_scalar, minimize
from scipy.special import expit as sigmoid
from sklearn.isotonic import IsotonicRegression

mpl.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "DejaVu Sans"],
})

labelsize = 16
titlesize = 18
legendsize = 13
ticksize = 13

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "tmp_latex_dir" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DECISION_TOKEN_IDX = 5

MODELS = {
    "Text": {
        "val_pred": ROOT / "results/final_sweep_v7_datasweepv3/optim_search_2026/bz32_lr1e-6_text/validation-ckpt-1322.jsonl",
        "test_pred": ROOT / "results/final_sweep_v7_datasweepv3/optim_search_2026/bz32_lr1e-6_text/finetuned-ckpt-1322.jsonl",
    },
    "Vision": {
        "val_pred": ROOT / "results/final_sweep_v7_datasweepv3/optim_search_2026/bz16_lr1e-6_vision/validation-ckpt-2648.jsonl",
        "test_pred": ROOT / "results/final_sweep_v7_datasweepv3/optim_search_2026/bz16_lr1e-6_vision/finetuned-ckpt-2648.jsonl",
    },
}

# Colors for methods
COLORS = {
    "Uncalibrated": "#888888",
    "Temperature": "#E74C3C",
    "Platt": "#3498DB",
    "Isotonic": "#2ECC71",
    "Histogram": "#F39C12",
}
LINESTYLES = {
    "Uncalibrated": "-",
    "Temperature": "--",
    "Platt": "-.",
    "Isotonic": ":",
    "Histogram": (0, (3, 1, 1, 1)),
}
LINEWIDTH = 2.5
MARKERSIZE = 7


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
    confidences, correct, logits = [], [], []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            pred = extract_prediction(rec["predict"])
            label = extract_prediction(rec["label"])
            if pred == "unknown":
                continue
            conf = math.exp(rec["token_logprobs"][DECISION_TOKEN_IDX])
            conf = np.clip(conf, 1e-7, 1 - 1e-7)
            logit = math.log(conf / (1 - conf))
            confidences.append(conf)
            logits.append(logit)
            correct.append(pred == label)
    return np.array(confidences), np.array(correct), np.array(logits)


def compute_ece(confidences, correct, n_bins=15):
    bin_edges = np.linspace(0.5, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences >= lo) & (confidences < hi) if hi < 1.0 else (confidences >= lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = correct[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += abs(bin_acc - bin_conf) * mask.sum() / len(confidences)
    return ece


# --- Calibration methods (same as calibration_posthoc.py) ---

def fit_temperature(val_logits, val_correct):
    def nll(T):
        scaled = val_logits / T
        p_pred = sigmoid(scaled)
        p_correct = np.where(val_correct, p_pred, 1 - p_pred)
        p_correct = np.clip(p_correct, 1e-10, 1.0)
        return -np.log(p_correct).mean()
    result = minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded")
    return result.x

def apply_temperature(logits, T):
    scaled = logits / T
    return sigmoid(np.abs(scaled))

def fit_platt(val_logits, val_correct):
    def nll(params):
        a, b = params
        scaled = a * val_logits + b
        p_pred = sigmoid(scaled)
        p_correct = np.where(val_correct, p_pred, 1 - p_pred)
        p_correct = np.clip(p_correct, 1e-10, 1.0)
        return -np.log(p_correct).mean()
    result = minimize(nll, x0=[1.0, 0.0], method="Nelder-Mead")
    return result.x

def apply_platt(logits, a, b):
    scaled = a * logits + b
    return sigmoid(np.abs(scaled))

def fit_isotonic(val_confidences, val_correct):
    ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    ir.fit(val_confidences, val_correct.astype(float))
    return ir

def apply_isotonic(confidences, ir):
    return ir.predict(confidences)

def fit_histogram_binning(val_confidences, val_correct, n_bins=15):
    bin_edges = np.linspace(0.5, 1.0, n_bins + 1)
    bin_accs = {}
    for i, (lo, hi) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        mask = (val_confidences >= lo) & (val_confidences < hi) if hi < 1.0 else (val_confidences >= lo) & (val_confidences <= hi)
        if mask.sum() > 0:
            bin_accs[i] = val_correct[mask].mean()
        else:
            bin_accs[i] = (lo + hi) / 2
    return bin_edges, bin_accs

def apply_histogram_binning(confidences, bin_edges, bin_accs):
    calibrated = np.zeros_like(confidences)
    for i, (lo, hi) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        mask = (confidences >= lo) & (confidences < hi) if hi < 1.0 else (confidences >= lo) & (confidences <= hi)
        calibrated[mask] = bin_accs[i]
    calibrated[calibrated == 0] = confidences[calibrated == 0]
    return calibrated


def panel_perbin(ax, method_confs, correct):
    """Per-bin reliability diagram with all methods overlaid."""
    bins = np.linspace(0.5, 1.0, 11)

    for method_name, confs in method_confs.items():
        accs, mean_confs = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (confs >= lo) & (confs < hi) if hi < 1.0 else (confs >= lo) & (confs <= hi)
            if mask.sum() > 0:
                accs.append(correct[mask].mean())
                mean_confs.append(confs[mask].mean())
            else:
                accs.append(np.nan)
                mean_confs.append(np.nan)
        ax.plot(mean_confs, accs, "o-",
                color=COLORS[method_name],
                linestyle=LINESTYLES[method_name],
                linewidth=LINEWIDTH, markersize=MARKERSIZE, alpha=0.85,
                label=method_name)

    ax.plot([0.5, 1.0], [0.5, 1.0], "-", color="grey", linewidth=1.5, alpha=0.5)
    ax.set_xlabel("Confidence", fontsize=labelsize)
    ax.set_ylabel("Accuracy", fontsize=labelsize)
    ax.set_xlim(0.48, 1.02)
    ax.set_ylim(0.45, 1.05)
    ax.tick_params(axis="both", labelsize=ticksize)
    ax.grid(True, linestyle="--", alpha=0.4)


def panel_cumulative(ax, method_confs, correct):
    """Cumulative accuracy (>= threshold) with all methods overlaid."""
    thresholds = np.linspace(0.5, 0.98, 100)

    for method_name, confs in method_confs.items():
        accs = []
        for t in thresholds:
            mask = confs >= t
            if mask.sum() > 0:
                accs.append(correct[mask].mean())
            else:
                accs.append(np.nan)
        ax.plot(thresholds, accs, color=COLORS[method_name],
                linestyle=LINESTYLES[method_name],
                linewidth=LINEWIDTH, alpha=0.85, label=method_name)

    ax.plot([0.5, 1.0], [0.5, 1.0], "-", color="grey", linewidth=1.5, alpha=0.5)
    ax.set_xlabel("Confidence Threshold", fontsize=labelsize)
    ax.set_ylabel("Accuracy (≥ threshold)", fontsize=labelsize)
    ax.set_xlim(0.48, 1.02)
    ax.set_ylim(0.45, 1.05)
    ax.tick_params(axis="both", labelsize=ticksize)
    ax.grid(True, linestyle="--", alpha=0.4)


def main():
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    for col_idx, (model_name, paths) in enumerate(MODELS.items()):
        # Load data
        val_conf, val_corr, val_logits = load_predictions(paths["val_pred"])
        test_conf, test_corr, test_logits = load_predictions(paths["test_pred"])

        print(f"\n{model_name}: val={len(val_conf)}, test={len(test_conf)}")

        # Fit all methods on validation
        T = fit_temperature(val_logits, val_corr)
        a, b = fit_platt(val_logits, val_corr)
        ir = fit_isotonic(val_conf, val_corr)
        bin_edges, bin_accs = fit_histogram_binning(val_conf, val_corr)

        # Apply to test
        method_confs = {
            "Uncalibrated": test_conf,
            "Temperature": apply_temperature(test_logits, T),
            "Platt": apply_platt(test_logits, a, b),
            "Isotonic": apply_isotonic(test_conf, ir),
            "Histogram": apply_histogram_binning(test_conf, bin_edges, bin_accs),
        }

        # Print ECEs
        for name, confs in method_confs.items():
            ece = compute_ece(confs, test_corr)
            print(f"  {name}: test ECE={ece:.4f}")

        # Plot
        panel_perbin(axes[0, col_idx], method_confs, test_corr)
        panel_cumulative(axes[1, col_idx], method_confs, test_corr)

        axes[0, col_idx].set_title(f"{model_name}: Per-Bin Reliability",
                                    fontsize=titlesize, fontweight="bold", pad=10)
        axes[1, col_idx].set_title(f"{model_name}: Cumulative Accuracy",
                                    fontsize=titlesize, fontweight="bold", pad=10)

    # Shared legend at top
    legend_handles = [
        Line2D([0], [0], color=COLORS[m], linestyle=LINESTYLES[m],
               linewidth=LINEWIDTH, marker="o" if m != "Uncalibrated" else None,
               markersize=MARKERSIZE, label=m)
        for m in ["Uncalibrated", "Temperature", "Platt", "Isotonic", "Histogram"]
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=5,
               fontsize=legendsize + 1, frameon=False, bbox_to_anchor=(0.5, 1.01))

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out = OUTPUT_DIR / "calibration_posthoc"
    plt.savefig(f"{out}.pdf", dpi=200, bbox_inches="tight")
    plt.savefig(f"{out}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out}.pdf")
    print(f"Saved: {out}.png")


if __name__ == "__main__":
    main()
