#!/usr/bin/env python3
"""
Post-hoc calibration comparison for SFT models.

Fits 4 methods on validation set, evaluates on test set:
  1. Temperature Scaling (Guo et al. 2017)
  2. Platt Scaling (Platt 1999)
  3. Isotonic Regression
  4. Histogram Binning (Zadrozny & Elkan 2001)

Usage: uv run scripts/calibration_posthoc.py
"""

import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar, minimize
from scipy.special import expit as sigmoid
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parent.parent

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
    """Load predictions, return (confidences, correct, logits).

    confidence: P(predicted class) = exp(logprob[5])
    logit: log(p / (1-p)) — logit of the confidence in the predicted class
    correct: bool array — whether prediction matches label
    """
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
    """Expected Calibration Error."""
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


# =============================================================================
# Method 1: Temperature Scaling
# =============================================================================
def fit_temperature(val_logits, val_correct):
    """Fit temperature T to minimize NLL on validation set."""
    def nll(T):
        # Scale logits by temperature
        scaled = val_logits / T
        # P(correct) = sigmoid(scaled) if correct, else 1 - sigmoid(scaled)
        p_pred = sigmoid(scaled)
        p_correct = np.where(val_correct, p_pred, 1 - p_pred)
        p_correct = np.clip(p_correct, 1e-10, 1.0)
        return -np.log(p_correct).mean()

    result = minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded")
    return result.x


def apply_temperature(logits, T):
    """Apply temperature scaling, return calibrated confidences."""
    scaled = logits / T
    return sigmoid(np.abs(scaled))  # confidence in predicted class


# =============================================================================
# Method 2: Platt Scaling
# =============================================================================
def fit_platt(val_logits, val_correct):
    """Fit Platt scaling: p = sigmoid(a * logit + b)."""
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
    """Apply Platt scaling, return calibrated confidences."""
    scaled = a * logits + b
    return sigmoid(np.abs(scaled))


# =============================================================================
# Method 3: Isotonic Regression
# =============================================================================
def fit_isotonic(val_confidences, val_correct):
    """Fit isotonic regression: monotonic mapping from confidence -> calibrated prob."""
    ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    ir.fit(val_confidences, val_correct.astype(float))
    return ir


def apply_isotonic(confidences, ir):
    """Apply isotonic regression."""
    return ir.predict(confidences)


# =============================================================================
# Method 4: Histogram Binning
# =============================================================================
def fit_histogram_binning(val_confidences, val_correct, n_bins=15):
    """Fit histogram binning: assign each bin its empirical accuracy."""
    bin_edges = np.linspace(0.5, 1.0, n_bins + 1)
    bin_accs = {}
    for i, (lo, hi) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        mask = (val_confidences >= lo) & (val_confidences < hi) if hi < 1.0 else (val_confidences >= lo) & (val_confidences <= hi)
        if mask.sum() > 0:
            bin_accs[i] = val_correct[mask].mean()
        else:
            bin_accs[i] = (lo + hi) / 2  # default to midpoint
    return bin_edges, bin_accs


def apply_histogram_binning(confidences, bin_edges, bin_accs):
    """Apply histogram binning."""
    calibrated = np.zeros_like(confidences)
    for i, (lo, hi) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        mask = (confidences >= lo) & (confidences < hi) if hi < 1.0 else (confidences >= lo) & (confidences <= hi)
        calibrated[mask] = bin_accs[i]
    # Handle any below 0.5 (shouldn't happen but just in case)
    calibrated[calibrated == 0] = confidences[calibrated == 0]
    return calibrated


# =============================================================================
# Main
# =============================================================================
def main():
    for model_name, paths in MODELS.items():
        print(f"\n{'='*60}")
        print(f"  {model_name} Model")
        print(f"{'='*60}")

        # Load data
        val_conf, val_corr, val_logits = load_predictions(paths["val_pred"])
        test_conf, test_corr, test_logits = load_predictions(paths["test_pred"])

        print(f"  Validation: {len(val_conf)} samples, acc={val_corr.mean():.4f}")
        print(f"  Test:       {len(test_conf)} samples, acc={test_corr.mean():.4f}")

        # Uncalibrated ECE
        val_ece_uncal = compute_ece(val_conf, val_corr)
        test_ece_uncal = compute_ece(test_conf, test_corr)
        print(f"\n  Uncalibrated:  val ECE={val_ece_uncal:.4f}, test ECE={test_ece_uncal:.4f}")

        results = []

        # 1. Temperature Scaling
        T = fit_temperature(val_logits, val_corr)
        val_cal = apply_temperature(val_logits, T)
        test_cal = apply_temperature(test_logits, T)
        val_ece = compute_ece(val_cal, val_corr)
        test_ece = compute_ece(test_cal, test_corr)
        print(f"  Temperature (T={T:.3f}): val ECE={val_ece:.4f}, test ECE={test_ece:.4f}")
        results.append(("Temperature", T, val_ece, test_ece, test_cal))

        # 2. Platt Scaling
        a, b = fit_platt(val_logits, val_corr)
        val_cal = apply_platt(val_logits, a, b)
        test_cal = apply_platt(test_logits, a, b)
        val_ece = compute_ece(val_cal, val_corr)
        test_ece = compute_ece(test_cal, test_corr)
        print(f"  Platt (a={a:.3f}, b={b:.3f}): val ECE={val_ece:.4f}, test ECE={test_ece:.4f}")
        results.append(("Platt", (a, b), val_ece, test_ece, test_cal))

        # 3. Isotonic Regression
        ir = fit_isotonic(val_conf, val_corr)
        val_cal = apply_isotonic(val_conf, ir)
        test_cal = apply_isotonic(test_conf, ir)
        val_ece = compute_ece(val_cal, val_corr)
        test_ece = compute_ece(test_cal, test_corr)
        print(f"  Isotonic:      val ECE={val_ece:.4f}, test ECE={test_ece:.4f}")
        results.append(("Isotonic", ir, val_ece, test_ece, test_cal))

        # 4. Histogram Binning
        bin_edges, bin_accs = fit_histogram_binning(val_conf, val_corr)
        val_cal = apply_histogram_binning(val_conf, bin_edges, bin_accs)
        test_cal = apply_histogram_binning(test_conf, bin_edges, bin_accs)
        val_ece = compute_ece(val_cal, val_corr)
        test_ece = compute_ece(test_cal, test_corr)
        print(f"  Histogram:     val ECE={val_ece:.4f}, test ECE={test_ece:.4f}")
        results.append(("Histogram", (bin_edges, bin_accs), val_ece, test_ece, test_cal))

        # Summary table
        print(f"\n  {'Method':<15} {'Val ECE':>10} {'Test ECE':>10} {'ΔECE':>10}")
        print(f"  {'-'*45}")
        print(f"  {'Uncalibrated':<15} {val_ece_uncal:>10.4f} {test_ece_uncal:>10.4f} {'—':>10}")
        for name, _, v_ece, t_ece, _ in results:
            delta = t_ece - test_ece_uncal
            print(f"  {name:<15} {v_ece:>10.4f} {t_ece:>10.4f} {delta:>+10.4f}")


if __name__ == "__main__":
    main()
