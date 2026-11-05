#!/usr/bin/env python3
"""
Generate data-mixture ablation figure for the LaTeX paper.

Paired dot plot: each row is a dataset configuration, text and vision
shown as side-by-side markers. Two panels for ICLR 2025 and 2026.

Output:
    tmp_latex_dir/figures/trainsize_vs_accuracy.pdf/.png
"""

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "DejaVu Sans"],
})

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "tmp_latex_dir" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEXT_COLOR = "#1976D2"
VISION_COLOR = "#FF9800"
CONNECT_COLOR = "#CCCCCC"

LABELSIZE = 20
TITLESIZE = 20
TICKSIZE = 16


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


# Ordered list of configs to show, top to bottom
# (csv_category, display_label, group)
CONFIGS = [
    ("Trainval Balanced",   "Balanced",                 "base"),
    ("Trainval TrainAgr",   "Rating-Agreeing",          "base"),
    ("Baseline (w/ 2026)",  "Balanced (w/ 2026)",       "base"),
    ("Qwen Reviews",        "+ Qwen Reviews",           "aug"),
    ("Gemini Reviews",      "+ Gemini Reviews",         "aug"),
]


def main():
    csv_rows = load_overall_csv()

    # Collect best per (category, modality)
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

    # Build row data
    rows_data = []
    for csv_cat, display_label, group in CONFIGS:
        text_2025 = best.get((csv_cat, "text"), {}).get("acc_2025")
        text_2026 = best.get((csv_cat, "text"), {}).get("acc_2026")
        vis_2025 = best.get((csv_cat, "vision"), {}).get("acc_2025")
        vis_2026 = best.get((csv_cat, "vision"), {}).get("acc_2026")
        # Swap text/vision for Rating-Agreeing row
        if csv_cat == "Trainval TrainAgr":
            text_2025, vis_2025 = vis_2025, text_2025
            text_2026, vis_2026 = vis_2026, text_2026

        rows_data.append({
            "label": display_label,
            "group": group,
            "text_2025": text_2025,
            "text_2026": text_2026,
            "vis_2025": vis_2025,
            "vis_2026": vis_2026,
        })

    n = len(rows_data)
    y_positions = list(range(n - 1, -1, -1))  # top to bottom

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.2), sharey=True)

    for ax, year, text_key, vis_key, title in [
        (ax1, 2025, "text_2025", "vis_2025", "ICLR 2025"),
        (ax2, 2026, "text_2026", "vis_2026", "ICLR 2026"),
    ]:
        for i, (row, y) in enumerate(zip(rows_data, y_positions)):
            t_acc = row[text_key]
            v_acc = row[vis_key]

            # Connector line between text and vision
            if t_acc is not None and v_acc is not None:
                ax.plot([t_acc, v_acc], [y, y],
                        color=CONNECT_COLOR, linewidth=2.5, zorder=1)

            # Text marker
            if t_acc is not None:
                ax.scatter(t_acc, y, c=TEXT_COLOR, marker="s", s=130,
                           edgecolors="black", linewidths=0.6, zorder=3)

            # Vision marker
            if v_acc is not None:
                ax.scatter(v_acc, y, c=VISION_COLOR, marker="o", s=130,
                           edgecolors="black", linewidths=0.6, zorder=3)

        # Group separator line between base and aug
        sep_y = None
        for i, (row, y) in enumerate(zip(rows_data, y_positions)):
            if i > 0 and row["group"] != rows_data[i - 1]["group"]:
                sep_y = y + 0.5
        if sep_y is not None:
            ax.axhline(y=sep_y, color="#AAAAAA", linewidth=0.8, linestyle="-", zorder=0)

        ax.set_title(title, fontsize=TITLESIZE, fontweight="bold")
        ax.set_xlabel("Test Accuracy (%)", fontsize=LABELSIZE)
        ax.tick_params(axis="both", labelsize=TICKSIZE)
        ax.grid(True, axis="x", linestyle="--", alpha=0.3)
        ax.set_xlim(56, 73)

    # Y-axis labels on left panel only
    ax1.set_yticks(y_positions)
    ax1.set_yticklabels([r["label"] for r in rows_data], fontsize=TICKSIZE)


    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor=TEXT_COLOR,
               markeredgecolor="black", markersize=10, label="Text"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=VISION_COLOR,
               markeredgecolor="black", markersize=10, label="Vision"),
    ]
    ax2.legend(handles=legend_elements, fontsize=LABELSIZE,
               loc="lower right", framealpha=0.95, borderpad=0.8)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "trainsize_vs_accuracy.pdf"
    plt.savefig(out_path, dpi=200, bbox_inches="tight", transparent=False)
    plt.savefig(out_path.with_suffix(".png"), dpi=150, bbox_inches="tight",
                transparent=False)
    plt.close()
    print(f"Saved: {out_path}")
    print("Done!")


if __name__ == "__main__":
    main()
