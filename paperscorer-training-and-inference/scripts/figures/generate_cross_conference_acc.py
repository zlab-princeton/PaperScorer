#!/usr/bin/env python3
"""
Generate single radar chart: SFT Text vs Midtrain+SFT — Accuracy only.

Output:
    tmp_latex_dir/figures/cross_conference_acc.pdf / .png
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "DejaVu Sans"],
})

labelsize = 22
titlesize = 26
legendsize = 20
ticksize = 18

LINEWIDTH = 3.5
MARKERSIZE = 12

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "tmp_latex_dir" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_DIR = ROOT / "results" / "balanced_nips_icml_eval"
TEXT_META = ROOT / "data" / "balanced_nips_icml_eval" / "data.json"

# ---------------------------------------------------------------------------
# Axes & models
# ---------------------------------------------------------------------------
AXES_KEYS = [("nips", 2025), ("ICML", 2025), ("nips", 2024)]
AXES_LABELS = ["NeurIPS '25", "ICML '25", "NeurIPS '24"]

MODELS = [
    ("SFT Text",
     RESULTS_DIR / "sft_text_bz32_ep2" / "finetuned-ckpt-1322.jsonl",
     TEXT_META),
    ("Midtrain + SFT",
     RESULTS_DIR / "pt_base_lr4e-6_ep1" / "finetuned-ckpt-661.jsonl",
     TEXT_META),
]

COLORS = ["#888888", "#5BAA5B"]
LINESTYLES = ["--", "-"]
FILL_ALPHAS = [0.08, 0.15]

# ---------------------------------------------------------------------------
# Compute metrics
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


def normalize_label(raw: str) -> str:
    return "reject" if raw.lower() == "reject" else "accept"


def compute_metrics(pred_path: Path, meta_path: Path):
    with open(meta_path) as f:
        meta = json.load(f)
    with open(pred_path) as f:
        preds = [json.loads(line) for line in f]
    assert len(preds) == len(meta)

    groups = defaultdict(lambda: {"correct": 0, "n": 0, "nA": 0, "nR": 0})
    for pred_rec, meta_rec in zip(preds, meta):
        m = meta_rec["_metadata"]
        key = (m["conference"], m["year"])
        pred = extract_prediction(pred_rec["predict"])
        label = normalize_label(m.get("answer", m.get("decision", "")))
        g = groups[key]
        g["n"] += 1
        if label == "accept":
            g["nA"] += 1
        else:
            g["nR"] += 1
        if pred == label:
            g["correct"] += 1

    result = {}
    for key, g in groups.items():
        result[key] = {
            "acc": g["correct"] / g["n"] * 100 if g["n"] else 0,
            "n": g["n"], "nA": g["nA"], "nR": g["nR"],
        }
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    all_data = {}
    for name, pred_path, meta_path in MODELS:
        all_data[name] = compute_metrics(pred_path, meta_path)

    model_names = [n for n, _, _ in MODELS]

    # Print summary
    print("\n=== Cross-Conference Accuracy ===\n")
    print(f"{'Model':<20} {'Venue':<14} {'Acc':>6}  (n)")
    print("-" * 50)
    for name in model_names:
        for key in AXES_KEYS:
            c = all_data[name].get(key, {})
            print(f"  {name:<18} {key[0]} {key[1]:<6} "
                  f"{c.get('acc',0):5.1f}%  (n={c.get('n',0)})")
        print()

    # Radar setup
    N = len(AXES_KEYS)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    all_vals = []
    for name in model_names:
        for key in AXES_KEYS:
            v = all_data[name].get(key, {}).get("acc")
            if v is not None:
                all_vals.append(v)
    ymin = int(min(all_vals)) - 12
    ymax = int(max(all_vals)) + 6

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "polar"})

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(ymin, ymax)

    yticks = list(range(int(ymin) + (5 - int(ymin) % 5) % 5, int(ymax) + 1, 5))
    ax.set_yticks(yticks)
    ax.set_yticklabels([""] * len(yticks))
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.xaxis.grid(True, linestyle="--", alpha=0.3)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(AXES_LABELS, fontsize=labelsize, fontweight="bold")
    ax.tick_params(axis="x", pad=30)

    # Compute offsets for annotations
    def _offsets(n):
        out, inn = [], []
        for i in range(n):
            angle = np.pi / 2 - (2 * np.pi * i / n)
            dx, dy = np.cos(angle), np.sin(angle)
            outer_mag = 40
            inner_mag = 75  # push grey labels well inside
            out.append((dx * outer_mag, dy * outer_mag))
            inn.append((-dx * inner_mag, -dy * inner_mag))
        return out, inn

    outer_off, inner_off = _offsets(N)

    values_list = []
    for name in model_names:
        vals = [all_data[name].get(key, {}).get("acc", ymin) for key in AXES_KEYS]
        values_list.append(vals)

    for idx, (name, color, ls, fa) in enumerate(
        zip(model_names, COLORS, LINESTYLES, FILL_ALPHAS)
    ):
        vals = values_list[idx]
        closed = vals + vals[:1]
        ax.plot(angles, closed, linestyle=ls, linewidth=LINEWIDTH,
                color=color, marker="o", markersize=MARKERSIZE, label=name,
                zorder=5)
        ax.fill(angles, closed, color=color, alpha=fa, zorder=2)

        for i, (angle, val) in enumerate(zip(angles[:-1], vals)):
            is_outer = all(val >= vl[i] for vl in values_list)
            dx, dy = outer_off[i] if is_outer else inner_off[i]
            ax.annotate(
                f"{val:.1f}%",
                xy=(angle, val), xycoords="data",
                xytext=(dx, dy), textcoords="offset points",
                ha="center", va="center",
                fontsize=ticksize, fontweight="bold", color=color,
            )

    ax.set_title("Cross-Conference Accuracy", fontsize=titlesize,
                 fontweight="bold", pad=120)

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18),
              ncol=2, fontsize=legendsize - 2, frameon=True, fancybox=True,
              edgecolor="#CCCCCC")

    plt.tight_layout()

    out = OUTPUT_DIR / "cross_conference_acc"
    plt.savefig(out.with_suffix(".pdf"), dpi=200, bbox_inches="tight",
                transparent=False)
    plt.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight",
                transparent=False)
    plt.close()
    print(f"Saved: {out.with_suffix('.pdf')}")
    print(f"Saved: {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
