#!/usr/bin/env python3
"""
Per-bin reliability diagram showing uncalibrated vs all 4 post-hoc corrections.
Fit on validation, applied to test. No coverage axis.

Output: calibration_val_correction.png/.pdf
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

labelsize = 20
titlesize = 22
legendsize = 16
ticksize = 16

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

METHODS = {
    "Uncalibrated": {"color": "#888888", "ls": "-", "lw": 3.0, "marker": "o", "ms": 10, "zorder": 5},
    "Temperature":  {"color": "#E74C3C", "ls": "--", "lw": 2.5, "marker": "s", "ms": 8, "zorder": 4},
    "Platt":        {"color": "#3498DB", "ls": "-.", "lw": 2.5, "marker": "D", "ms": 8, "zorder": 4},
}


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
        ece += abs(correct[mask].mean() - confidences[mask].mean()) * mask.sum() / len(confidences)
    return ece


# --- Calibration methods ---

def fit_temperature(val_logits, val_correct):
    def nll(T):
        p = sigmoid(val_logits / T)
        pc = np.where(val_correct, p, 1 - p)
        return -np.log(np.clip(pc, 1e-10, 1.0)).mean()
    return minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded").x

def apply_temperature(logits, T):
    return sigmoid(np.abs(logits / T))

def fit_platt(val_logits, val_correct):
    def nll(params):
        a, b = params
        p = sigmoid(a * val_logits + b)
        pc = np.where(val_correct, p, 1 - p)
        return -np.log(np.clip(pc, 1e-10, 1.0)).mean()
    return minimize(nll, x0=[1.0, 0.0], method="Nelder-Mead").x

def apply_platt(logits, a, b):
    return sigmoid(np.abs(a * logits + b))

def fit_isotonic(val_conf, val_correct):
    ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    ir.fit(val_conf, val_correct.astype(float))
    return ir

def apply_isotonic(conf, ir):
    return ir.predict(conf)

def fit_histogram(val_conf, val_correct, n_bins=15):
    edges = np.linspace(0.5, 1.0, n_bins + 1)
    accs = {}
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (val_conf >= lo) & (val_conf < hi) if hi < 1.0 else (val_conf >= lo) & (val_conf <= hi)
        accs[i] = val_correct[mask].mean() if mask.sum() > 0 else (lo + hi) / 2
    return edges, accs

def apply_histogram(conf, edges, accs):
    cal = np.zeros_like(conf)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (conf >= lo) & (conf < hi) if hi < 1.0 else (conf >= lo) & (conf <= hi)
        cal[mask] = accs[i]
    cal[cal == 0] = conf[cal == 0]
    return cal


def plot_reliability(ax, method_confs, correct, title):
    """Per-bin reliability diagram, all methods overlaid."""
    bins = np.linspace(0.5, 1.0, 11)

    for method_name, confs in method_confs.items():
        style = METHODS[method_name]
        accs, mean_confs = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (confs >= lo) & (confs < hi) if hi < 1.0 else (confs >= lo) & (confs <= hi)
            if mask.sum() > 0:
                accs.append(correct[mask].mean())
                mean_confs.append(confs[mask].mean())
            else:
                accs.append(np.nan)
                mean_confs.append(np.nan)

        ece = compute_ece(confs, correct)
        label = f"{method_name} (ECE={ece:.3f})"
        ax.plot(mean_confs, accs, marker=style["marker"], linestyle=style["ls"],
                color=style["color"], linewidth=style["lw"], markersize=style["ms"],
                alpha=0.9, zorder=style["zorder"], label=label)

    # Perfect calibration line
    ax.plot([0.5, 1.0], [0.5, 1.0], "-", color="grey", linewidth=1.5, alpha=0.5)
    ax.set_xlabel("Confidence", fontsize=labelsize)
    ax.set_ylabel("Accuracy", fontsize=labelsize)
    ax.set_xlim(0.48, 1.02)
    ax.set_ylim(0.45, 1.05)
    ax.set_title(title, fontsize=titlesize, fontweight="bold", pad=12)
    ax.tick_params(axis="both", labelsize=ticksize)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=legendsize - 2, loc="upper left")

    for t_val in [0.7, 0.8, 0.9]:
        ax.axvline(x=t_val, color="#BBBBBB", linewidth=1.0, linestyle=":", alpha=0.6, zorder=0)


def main():
    available = {}
    for model_name, paths in MODELS.items():
        if paths["val_pred"].exists() and paths["test_pred"].exists():
            available[model_name] = paths

    n = len(available)
    if n == 0:
        print("No validation predictions found yet.")
        return

    fig, axes = plt.subplots(1, n, figsize=(10 * n, 8))
    if n == 1:
        axes = [axes]

    for idx, (model_name, paths) in enumerate(available.items()):
        val_conf, val_corr, val_logits = load_predictions(paths["val_pred"])
        test_conf, test_corr, test_logits = load_predictions(paths["test_pred"])

        print(f"{model_name}: val={len(val_conf)}, test={len(test_conf)}")

        # Fit on val
        T = fit_temperature(val_logits, val_corr)
        a, b = fit_platt(val_logits, val_corr)

        print(f"  Temperature T={T:.3f}")
        print(f"  Platt a={a:.3f}, b={b:.3f}")

        # Apply to test
        method_confs = {
            "Uncalibrated": test_conf,
            "Temperature": apply_temperature(test_logits, T),
            "Platt": apply_platt(test_logits, a, b),
        }

        for name, confs in method_confs.items():
            ece = compute_ece(confs, test_corr)
            print(f"  {name}: test ECE={ece:.4f}")

        plot_reliability(axes[idx], method_confs, test_corr, model_name)

    plt.tight_layout()

    out = OUTPUT_DIR / "calibration_val_correction"
    plt.savefig(f"{out}.pdf", dpi=200, bbox_inches="tight")
    plt.savefig(f"{out}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out}.pdf / .png")


if __name__ == "__main__":
    main()
