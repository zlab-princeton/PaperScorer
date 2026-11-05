"""Stage C: assemble the 10-model x 58-paper grid into tables + plots.

Reads logs/grid/scores/<model>.jsonl (from score_from_cache), produces:
  grid/scores_long.csv                     model,paper_id,group,order,subject,p_accept,decision
  grid/heatmap_<group>.png                 papers x models, cell=p_accept (anon_gt, our_lab, iclr_best_paper)
  grid/commit_trajectory_allmodels.png     10 lines (one per model), p_accept vs commit order
  grid/summary.txt                         per-model accept-rate + the neurips anchor row

Usage:
  python assemble_grid.py --scores-dir logs/grid/scores --models-tsv scripts/grid_models.tsv \
      --out-dir logs/grid
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams.update({"text.usetex": False, "font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "DejaVu Sans"]})


def load_models(tsv: Path) -> list[str]:
    out = []
    for line in tsv.read_text().splitlines():
        if line.strip():
            out.append(line.split("\t")[0])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores-dir", required=True)
    ap.add_argument("--models-tsv", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    scores_dir = Path(args.scores_dir)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    model_order = load_models(Path(args.models_tsv))
    models = [m for m in model_order if (scores_dir / f"{m}.jsonl").exists()]
    print(f"[assemble] {len(models)}/{len(model_order)} model score files present")

    # rec[(model, paper_id)] = dict; paper_meta[paper_id] = {group, order, subject}
    rec, pmeta = {}, {}
    for m in models:
        for line in (scores_dir / f"{m}.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("state") != "done":
                continue
            rec[(m, r["paper_id"])] = r
            pmeta.setdefault(r["paper_id"], {"group": r.get("group"), "order": r.get("order"),
                                             "subject": r.get("subject")})

    # long CSV
    with (out / "scores_long.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["model", "paper_id", "group", "order", "subject", "p_accept", "decision"])
        for (m, pid), r in sorted(rec.items()):
            w.writerow([m, pid, r.get("group"), r.get("order"), r.get("subject"), r.get("p_accept"), r.get("decision")])
    print(f"[assemble] wrote scores_long.csv ({len(rec)} cells)")

    # ---- heatmaps per discrete group ----
    def heatmap(group, fname, label_key="subject", trunc=40):
        pids = sorted([p for p, mm in pmeta.items() if mm["group"] == group],
                      key=lambda p: pmeta[p].get("order") or 0)
        if not pids:
            return
        M = np.full((len(pids), len(models)), np.nan)
        for i, pid in enumerate(pids):
            for j, m in enumerate(models):
                r = rec.get((m, pid))
                if r and r.get("p_accept") is not None:
                    M[i, j] = r["p_accept"]
        fig, ax = plt.subplots(figsize=(max(8, 1.0 * len(models) + 4), max(4, 0.5 * len(pids) + 2)))
        im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(models))); ax.set_xticklabels(models, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(len(pids)))
        ax.set_yticklabels([(pmeta[p].get(label_key) or p)[:trunc] for p in pids], fontsize=9)
        for i in range(len(pids)):
            for j in range(len(models)):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=7,
                            color="black")
        ax.set_title(f"PaperLens p_accept — {group}  ({len(pids)} papers x {len(models)} models)", fontsize=13, pad=12)
        fig.colorbar(im, ax=ax, fraction=0.025, label="p_accept")
        plt.tight_layout(); plt.savefig(out / fname, dpi=170, bbox_inches="tight"); plt.close()
        print(f"[assemble] -> {fname}")

    for g in ["anon_gt", "our_lab", "iclr_best_paper", "anon_ones_we_got_right"]:
        heatmap(g, f"heatmap_{g}.png")

    # ---- commit trajectory: 10 lines ----
    commit_pids = sorted([p for p, mm in pmeta.items() if mm["group"] == "commit"],
                         key=lambda p: pmeta[p]["order"])
    if commit_pids:
        fig, ax = plt.subplots(figsize=(max(10, 0.5 * len(commit_pids) + 4), 6))
        xs = list(range(len(commit_pids)))
        cmap = plt.get_cmap("tab10")
        for j, m in enumerate(models):
            ys = [rec[(m, p)]["p_accept"] if (m, p) in rec else np.nan for p in commit_pids]
            ax.plot(xs, ys, "-o", ms=3, lw=1.2, color=cmap(j % 10), label=m, alpha=0.85)
        ax.axhline(0.5, color="black", ls="--", lw=1, alpha=0.5)
        ax.set_ylim(0, 1.02); ax.set_ylabel("p_accept", fontsize=14)
        ax.set_title("Commit trajectory across all models (zlab template; relative, not absolute)", fontsize=12, pad=10)
        ax.set_xticks(xs)
        ax.set_xticklabels([(pmeta[p]["subject"] or p)[:28] for p in commit_pids], rotation=45, ha="right", fontsize=7)
        ax.legend(fontsize=8, ncol=2, loc="upper left")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        plt.tight_layout(); plt.savefig(out / "commit_trajectory_allmodels.png", dpi=170, bbox_inches="tight"); plt.close()
        print("[assemble] -> commit_trajectory_allmodels.png")

    # ---- summary ----
    with (out / "summary.txt").open("w") as f:
        f.write("Per-model accept rate (over all scored papers):\n")
        for m in models:
            ps = [r["p_accept"] for (mm, _), r in rec.items() if mm == m and r.get("p_accept") is not None]
            if ps:
                acc = sum(1 for p in ps if p >= 0.5) / len(ps)
                f.write(f"  {m:<20} n={len(ps):<3} accept_rate={acc:.2f} mean_p={np.mean(ps):.3f}\n")
        # neurips anchor
        f.write("\nNeurIPS/COLM submission (neurips) per model:\n")
        for m in models:
            r = rec.get((m, "neurips"))
            if r:
                f.write(f"  {m:<20} p_accept={r['p_accept']:.3f} {r['decision']}\n")
    print((out / "summary.txt").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
