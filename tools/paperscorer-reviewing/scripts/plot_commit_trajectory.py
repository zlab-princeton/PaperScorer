"""Plot PaperLens p_accept across commits (chronological).

Reads results.jsonl from score_commits.py; renders a line+marker trajectory
with x = commit order (short sha + truncated subject), Accept/Reject coloring,
0.5 threshold line. Lab matplotlib style.

Usage:
  python plot_commit_trajectory.py --results logs/commit_track/results.jsonl \
      --out logs/commit_track/trajectory.png
"""
import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
})
ACCEPT = "#1a7f37"
REJECT = "#cf222e"
LINE = "#1f6feb"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="PaperLens p_accept across commits (lab template; trajectory, not absolute)")
    ap.add_argument("--reference-jsonl", default=None,
                    help="optional results_<label>.jsonl (single row) to draw as a horizontal reference line, e.g. the NeurIPS/COLM submission")
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.results).read_text().splitlines() if l.strip()]
    rows.sort(key=lambda r: r.get("order", 0))
    done = [r for r in rows if r.get("state") == "done" and r.get("p_accept") is not None]
    if not done:
        print("no scored commits to plot"); return 1

    xs = list(range(len(done)))
    ys = [r["p_accept"] for r in done]
    labels = [f"{r['short']}\n{r['subject'][:32]}" for r in done]
    colors = [ACCEPT if y >= 0.5 else REJECT for y in ys]

    fig, ax = plt.subplots(figsize=(max(8, 1.3 * len(done)), 6))
    ax.plot(xs, ys, "-", color=LINE, linewidth=1.5, zorder=1, alpha=0.7)
    ax.scatter(xs, ys, c=colors, s=110, zorder=2, edgecolor="white", linewidth=1.5)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1, alpha=0.5)

    # Optional reference line (e.g. the NeurIPS/COLM submission version)
    if args.reference_jsonl and Path(args.reference_jsonl).exists():
        refs = [json.loads(l) for l in Path(args.reference_jsonl).read_text().splitlines() if l.strip()]
        ref = next((r for r in refs if r.get("state") == "done" and r.get("p_accept") is not None), None)
        if ref:
            rp = ref["p_accept"]; rl = ref.get("label", "reference")
            ax.axhline(rp, color="#8957e5", linestyle=":", linewidth=1.8, alpha=0.9)
            ax.text(len(xs) - 1, rp, f" {rl}: {rp:.2f}", color="#8957e5", va="bottom", ha="right", fontsize=11)

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("p_accept", fontsize=18)
    ax.set_title(args.title, fontsize=15, pad=16)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax.tick_params(axis="y", labelsize=12)
    for x, y in zip(xs, ys):
        ax.text(x, y + 0.02, f"{y:.2f}", ha="center", va="bottom", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(x=0.04)
    plt.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=180, bbox_inches="tight")
    print(f"-> {args.out}")

    # also dump a compact text summary
    print("\norder  short     date        p_accept  decision  subject")
    for r in done:
        print(f"  {r['order']:>2}   {r['short']}  {r['date']}   {r['p_accept']:.3f}    {r['decision']:<7} {r['subject'][:50]}")
    skipped = [r for r in rows if r.get("state") != "done"]
    if skipped:
        print(f"\nskipped/failed ({len(skipped)}):")
        for r in skipped:
            print(f"  {r['short']}  {r.get('state')}  {r.get('error','')[:80]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
