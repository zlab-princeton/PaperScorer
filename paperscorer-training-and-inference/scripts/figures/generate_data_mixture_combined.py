#!/usr/bin/env python3
"""
Combined figure: (a,b) data-mixture dot plots + (c) cross-conference bar chart.

Output:
    tmp_latex_dir/figures/data_mixture_combined.pdf / .png
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "helvetica",
})

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "tmp_latex_dir" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Lab style sizes ──
LABELSIZE = 28
TITLESIZE = 34
TICKSIZE = 20
LEGENDSIZE = 20

# ── Lab color palette ──
RED = "#FF8988"
ORANGE = "#FECC81"
BLUE = "#6098FF"
GREEN = "#77B25D"
PURPLE = "#B28CFF"

# ==========================================================================
# (a, b) Data-mixture dot plot
# ==========================================================================
TEXT_COLOR = BLUE
VISION_COLOR = ORANGE
CONNECT_COLOR = "#CCCCCC"

CONFIGS = [
    ("Trainval Balanced",   "'20--'23, '25",              "base"),
    ("Trainval TrainAgr",   "'20--'23, '25 (agreeing)",   "base"),
    ("Baseline (w/ 2026)",  "'20--'23, '25--'26",         "base"),
    ("Qwen Reviews",        "+ Qwen Reviews",              "aug"),
    ("Gemini Reviews",      "+ Gemini Reviews",            "aug"),
]


def load_overall_csv():
    csv_path = ROOT / "results" / "summarized_investigation" / "modality_v7" / "OVERALL.csv"
    rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def parse_accuracy(val):
    if not val or val.strip() == "":
        return None
    try:
        return float(val.strip())
    except ValueError:
        return None


def build_dot_data():
    csv_rows = load_overall_csv()
    all_cats = {c[0] for c in CONFIGS}
    best = {}
    for row in csv_rows:
        cat, modality = row["Category"], row["Type"]
        if cat not in all_cats or modality not in ("text", "vision"):
            continue
        acc_2025 = parse_accuracy(row.get("Best 2025 (%)", ""))
        acc_2026 = parse_accuracy(row.get("Best 2026 (%)", ""))
        key = (cat, modality)
        if key not in best:
            best[key] = {"acc_2025": acc_2025, "acc_2026": acc_2026}
        else:
            if acc_2025 is not None and (best[key]["acc_2025"] is None or acc_2025 > best[key]["acc_2025"]):
                best[key]["acc_2025"] = acc_2025
            if acc_2026 is not None and (best[key]["acc_2026"] is None or acc_2026 > best[key]["acc_2026"]):
                best[key]["acc_2026"] = acc_2026

    rows_data = []
    for csv_cat, display_label, group in CONFIGS:
        text_2025 = best.get((csv_cat, "text"), {}).get("acc_2025")
        text_2026 = best.get((csv_cat, "text"), {}).get("acc_2026")
        vis_2025 = best.get((csv_cat, "vision"), {}).get("acc_2025")
        vis_2026 = best.get((csv_cat, "vision"), {}).get("acc_2026")
        # No swap — keep text as text, vision as vision
        rows_data.append({
            "label": display_label, "group": group,
            "text_2025": text_2025, "text_2026": text_2026,
            "vis_2025": vis_2025, "vis_2026": vis_2026,
        })
    return rows_data


def draw_dot_panel(ax, rows_data, text_key, vis_key, title, show_ylabels=True,
                   show_xlabel=True):
    n = len(rows_data)
    y_positions = list(range(n - 1, -1, -1))

    for row, y in zip(rows_data, y_positions):
        t_acc, v_acc = row[text_key], row[vis_key]
        if t_acc is not None and v_acc is not None:
            ax.plot([t_acc, v_acc], [y, y],
                    color=CONNECT_COLOR, linewidth=2.5, zorder=1)
        if t_acc is not None:
            ax.scatter(t_acc, y, c=TEXT_COLOR, marker="s", s=130,
                       edgecolors="black", linewidths=1.2, zorder=3)
        if v_acc is not None:
            ax.scatter(v_acc, y, c=VISION_COLOR, marker="o", s=130,
                       edgecolors="black", linewidths=1.2, zorder=3)

    sep_y = None
    for i, (row, y) in enumerate(zip(rows_data, y_positions)):
        if i > 0 and row["group"] != rows_data[i - 1]["group"]:
            sep_y = y + 0.5
    if sep_y is not None:
        ax.axhline(y=sep_y, color="#AAAAAA", linewidth=0.8, linestyle="-", zorder=0)

    # Highlight the "'20--'23, '25--'26" row (index 2) as the final ReviewVL config
    reviewvl_idx = 2  # "Baseline (w/ 2026)" is the 3rd config
    reviewvl_y = y_positions[reviewvl_idx]
    ax.axhspan(reviewvl_y - 0.4, reviewvl_y + 0.4, color=GREEN, alpha=0.15, zorder=0)
    if show_ylabels:
        # Bold the y-tick label and add star
        labels = [r["label"] for r in rows_data]
        labels[reviewvl_idx] = r"\textbf{" + labels[reviewvl_idx] + r"} $\bigstar$"
        ax.set_yticklabels(labels, fontsize=TICKSIZE)

    ax.set_title(title, fontsize=TITLESIZE - 8, fontweight="bold")
    ax.tick_params(axis="both", labelsize=TICKSIZE)
    ax.grid(False)
    ax.set_xlim(56, 73)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)

    ax.set_yticks(y_positions)
    if show_ylabels:
        ax.set_yticklabels([r["label"] for r in rows_data], fontsize=TICKSIZE)
    else:
        ax.set_yticklabels([])

    if show_xlabel:
        ax.set_xlabel("Test Accuracy (\\%)", fontsize=LABELSIZE)
    else:
        plt.setp(ax.get_xticklabels(), visible=False)
        ax.set_xlabel("")


# ==========================================================================
# (c) Cross-conference accuracy
# ==========================================================================
RESULTS_DIR = ROOT / "results" / "balanced_nips_icml_eval"
TEXT_META = ROOT / "data" / "balanced_nips_icml_eval" / "data.json"

CROSS_CONF_KEYS = [("nips", 2025), ("ICML", 2025), ("nips", 2024)]
CROSS_CONF_LABELS = ["NeurIPS '25", "ICML '25", "NeurIPS '24"]

CROSS_CONF_MODELS = [
    ("SFT Text",
     RESULTS_DIR / "sft_text_bz32_ep2" / "finetuned-ckpt-1322.jsonl",
     TEXT_META),
    ("Midtrain + SFT",
     RESULTS_DIR / "pt_base_lr4e-6_ep1" / "finetuned-ckpt-661.jsonl",
     TEXT_META),
]

CROSS_CONF_COLORS = [RED, GREEN]


def extract_prediction(text):
    t = text.lower().strip()
    if "boxed{accept}" in t: return "accept"
    if "boxed{reject}" in t: return "reject"
    if "accept" in t: return "accept"
    if "reject" in t: return "reject"
    return "unknown"


def compute_cross_conf_metrics(pred_path, meta_path):
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
        label = "reject" if m.get("answer", m.get("decision", "")).lower() == "reject" else "accept"
        g = groups[key]
        g["n"] += 1
        if label == "accept": g["nA"] += 1
        else: g["nR"] += 1
        if pred == label: g["correct"] += 1

    result = {}
    for key, g in groups.items():
        result[key] = {
            "acc": g["correct"] / g["n"] * 100 if g["n"] else 0,
            "n": g["n"], "nA": g["nA"], "nR": g["nR"],
        }
    return result


def draw_cross_conf_bar(ax, all_data, model_names):
    x = np.arange(len(CROSS_CONF_KEYS))
    width = 0.34

    all_vals = []
    for name in model_names:
        for key in CROSS_CONF_KEYS:
            v = all_data[name].get(key, {}).get("acc")
            if v is not None:
                all_vals.append(v)

    ymin = max(0, np.floor(min(all_vals) - 2))
    ymax = min(100, np.ceil(max(all_vals) + 4))

    # Darker text colors for annotations (lab palette colors are light)
    text_colors = ["#CC4040", "#3D7A2E"]

    for idx, (name, color, tc) in enumerate(
        zip(model_names, CROSS_CONF_COLORS, text_colors)
    ):
        vals = [all_data[name].get(key, {}).get("acc", 0) for key in CROSS_CONF_KEYS]
        offsets = x + (idx - 0.5) * width
        bars = ax.bar(offsets, vals, width=width, color=color, label=name,
                      edgecolor="black", linewidth=0.8, zorder=3)

        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.5,
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=TICKSIZE,
                fontweight="bold",
                color=tc,
            )

    # Title handled at figure level for alignment
    ax.set_xticks(x)
    ax.set_xticklabels(CROSS_CONF_LABELS, fontsize=TICKSIZE)
    ax.set_ylabel("Accuracy (\\%)", fontsize=LABELSIZE)
    ax.set_ylim(ymin, ymax)
    ax.tick_params(axis="both", labelsize=TICKSIZE)
    ax.grid(False)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
    ax.legend(loc="upper right", fontsize=LEGENDSIZE, framealpha=0.95)


# ==========================================================================
# Main: combined figure
# ==========================================================================

def main():
    # Load data
    rows_data = build_dot_data()

    all_cross_conf = {}
    model_names = []
    for name, pred_path, meta_path in CROSS_CONF_MODELS:
        all_cross_conf[name] = compute_cross_conf_metrics(pred_path, meta_path)
        model_names.append(name)

    # Print cross-conference summary
    print("\n=== Cross-Conference Accuracy ===\n")
    for name in model_names:
        for key in CROSS_CONF_KEYS:
            c = all_cross_conf[name].get(key, {})
            print(f"  {name:<18} {key[0]} {key[1]:<6} "
                  f"{c.get('acc',0):5.1f}%  (n={c.get('n',0)})")
        print()

    # Create figure: 1×3 — two dot plots + cross-conference bar chart
    fig = plt.figure(figsize=(28, 8))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.2, 1.2, 1.3], wspace=0.38)

    ax1 = fig.add_subplot(gs[0, 0])  # (a) ICLR 2025 Test
    ax2 = fig.add_subplot(gs[0, 1])  # (b) ICLR 2026 Test
    ax3 = fig.add_subplot(gs[0, 2])  # (c) grouped bar chart

    # Draw dot panels
    draw_dot_panel(ax1, rows_data, "text_2025", "vis_2025", "(a) ICLR 2025 Test",
                   show_ylabels=True, show_xlabel=True)
    draw_dot_panel(ax2, rows_data, "text_2026", "vis_2026", "(b) ICLR 2026 Test",
                   show_ylabels=False, show_xlabel=True)

    # Aligned super-titles for both halves
    plt.tight_layout()
    left_center = (gs[0, 0].get_position(fig).x0 + gs[0, 1].get_position(fig).x1) / 2
    right_center = (gs[0, 2].get_position(fig).x0 + gs[0, 2].get_position(fig).x1) / 2
    title_y = 0.94
    fig.text(left_center, title_y, "Data Mixture", ha="center", va="bottom",
             fontsize=TITLESIZE + 2, fontweight="bold")
    fig.text(right_center, title_y, "(c) Cross-Conference Accuracy", ha="center",
             va="bottom", fontsize=TITLESIZE + 2, fontweight="bold")

    # Dot plot legend in panel (a)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor=TEXT_COLOR,
               markeredgecolor="black", markersize=10, label="Text"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=VISION_COLOR,
               markeredgecolor="black", markersize=10, label="Vision"),
    ]
    ax1.legend(handles=legend_elements, fontsize=LEGENDSIZE,
               loc="upper left", framealpha=0.95, borderpad=0.8)

    draw_cross_conf_bar(ax3, all_cross_conf, model_names)

    out = OUTPUT_DIR / "data_mixture_combined"
    plt.savefig(out.with_suffix(".pdf"), dpi=200, bbox_inches="tight",
                transparent=False)
    plt.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight",
                transparent=False)
    plt.close()
    print(f"Saved: {out.with_suffix('.pdf')}")
    print(f"Saved: {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
