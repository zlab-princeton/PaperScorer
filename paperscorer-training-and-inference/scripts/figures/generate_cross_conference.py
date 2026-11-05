#!/usr/bin/env python3
"""
Generate 1×3 radar chart: SFT Text vs PT Base lr4e-6 on balanced NeurIPS/ICML eval.

Panel (a): Accuracy
Panel (b): Accept Recall
Panel (c): Reject Recall

Each radar has 3 axes: NeurIPS '24, NeurIPS '25, ICML '25.
Two polygons per panel: SFT Text (grey dashed) vs PT Base (green solid).

Output:
    tmp_latex_dir/figures/cross_conference.pdf / .png
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

labelsize = 18
titlesize = 18
legendsize = 14
ticksize = 12

LINEWIDTH = 2.5
MARKERSIZE = 8

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
AXES_KEYS = [("nips", 2024), ("nips", 2025), ("ICML", 2025)]
AXES_LABELS = ["NeurIPS '24", "NeurIPS '25", "ICML '25"]

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

    groups = defaultdict(lambda: {"correct": 0, "n": 0,
                                   "a_correct": 0, "nA": 0,
                                   "r_correct": 0, "nR": 0})
    for pred_rec, meta_rec in zip(preds, meta):
        m = meta_rec["_metadata"]
        key = (m["conference"], m["year"])
        pred = extract_prediction(pred_rec["predict"])
        label = normalize_label(m.get("answer", m.get("decision", "")))
        g = groups[key]
        g["n"] += 1
        if pred == label:
            g["correct"] += 1
        if label == "accept":
            g["nA"] += 1
            if pred == "accept":
                g["a_correct"] += 1
        else:
            g["nR"] += 1
            if pred == "reject":
                g["r_correct"] += 1

    result = {}
    for key, g in groups.items():
        result[key] = {
            "acc": g["correct"] / g["n"] * 100 if g["n"] else 0,
            "ar": g["a_correct"] / g["nA"] * 100 if g["nA"] else None,
            "rr": g["r_correct"] / g["nR"] * 100 if g["nR"] else None,
            "n": g["n"], "nA": g["nA"], "nR": g["nR"],
        }
    return result


# ---------------------------------------------------------------------------
# Radar helper
# ---------------------------------------------------------------------------

def _offsets_for_n(n):
    out, inn = [], []
    for i in range(n):
        angle = np.pi / 2 - (2 * np.pi * i / n)
        dx, dy = np.cos(angle), np.sin(angle)
        mag = 20
        out.append((dx * mag, dy * mag))
        inn.append((-dx * mag, -dy * mag))
    return out, inn


def make_radar(ax, all_data, model_names, colors, linestyles, fill_alphas,
               title, axes_keys, axes_labels, metric, ymin, ymax):
    N = len(axes_keys)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(ymin, ymax)

    yticks = list(range(int(ymin) + (5 - int(ymin) % 5) % 5, int(ymax) + 1, 5))
    ax.set_yticks(yticks)
    ax.set_yticklabels([""] * len(yticks))
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.xaxis.grid(True, linestyle="--", alpha=0.3)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axes_labels, fontsize=labelsize - 2, fontweight="bold")

    values_list = []
    for name in model_names:
        vals = []
        for key in axes_keys:
            v = all_data[name].get(key, {}).get(metric)
            vals.append(v if v is not None else ymin)
        values_list.append(vals)

    outer_off, inner_off = _offsets_for_n(N)

    for idx, (name, color, ls, fa) in enumerate(
        zip(model_names, colors, linestyles, fill_alphas)
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

    ax.set_title(title, fontsize=titlesize, fontweight="bold", pad=25)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Load data
    all_data = {}
    for name, pred_path, meta_path in MODELS:
        all_data[name] = compute_metrics(pred_path, meta_path)

    model_names = [n for n, _, _ in MODELS]

    # Print summary
    print("\n=== Balanced Cross-Conference Results ===\n")
    print(f"{'Model':<20} {'Venue':<14} {'Acc':>6} {'AR':>6} {'RR':>6}  (n, A/R)")
    print("-" * 75)
    for name in model_names:
        for key in AXES_KEYS:
            c = all_data[name].get(key, {})
            print(f"  {name:<18} {key[0]} {key[1]:<6} "
                  f"{c.get('acc',0):5.1f}% {c.get('ar',0):5.1f}% {c.get('rr',0):5.1f}%  "
                  f"(n={c.get('n',0)}, {c.get('nA',0)}A/{c.get('nR',0)}R)")
        print()

    # Shared range across all 3 panels
    all_vals = []
    for name in model_names:
        for key in AXES_KEYS:
            for m in ("acc", "ar", "rr"):
                v = all_data[name].get(key, {}).get(m)
                if v is not None:
                    all_vals.append(v)
    ymin = max(45, (min(all_vals) // 10) * 10 - 5)
    ymax = min(95, (max(all_vals) // 10 + 1) * 10 + 5)

    # Build 1×3 figure — compact
    fig, (ax_acc, ax_ar, ax_rr) = plt.subplots(
        1, 3, figsize=(14, 5.5), subplot_kw={"projection": "polar"})

    make_radar(ax_acc, all_data, model_names, COLORS, LINESTYLES, FILL_ALPHAS,
               "(a) Accuracy", AXES_KEYS, AXES_LABELS, "acc", ymin, ymax)

    make_radar(ax_ar, all_data, model_names, COLORS, LINESTYLES, FILL_ALPHAS,
               "(b) Accept Recall", AXES_KEYS, AXES_LABELS, "ar", ymin, ymax)

    make_radar(ax_rr, all_data, model_names, COLORS, LINESTYLES, FILL_ALPHAS,
               "(c) Reject Recall", AXES_KEYS, AXES_LABELS, "rr", ymin, ymax)

    # Vertical legend at bottom
    handles, labels = ax_acc.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=1,
               fontsize=legendsize, frameon=True, fancybox=True,
               edgecolor="#CCCCCC", bbox_to_anchor=(0.5, -0.06))

    plt.tight_layout(rect=[0, 0.08, 1, 1])

    out = OUTPUT_DIR / "cross_conference"
    plt.savefig(out.with_suffix(".pdf"), dpi=200, bbox_inches="tight",
                transparent=False)
    plt.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight",
                transparent=False)
    plt.close()
    print(f"Saved: {out.with_suffix('.pdf')}")
    print(f"Saved: {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
