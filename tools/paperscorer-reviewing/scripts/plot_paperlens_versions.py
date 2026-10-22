"""PaperLens paper — p(Accept) across its whole lifetime, scored by the 4 7B models.

Chronological timeline (one continuous trajectory per model):
  COLM submission (2026-03-31) -> NeurIPS submission (2026-05-05)
  -> 28 arXiv-release commits (2026-05-11 .. 2026-05-26)

x-tick label per point = "MM/DD\n(version / commit message)".

Usage:
  python plot_paperlens_versions.py     # -> logs/grid/maps/paperlens_versions.png
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

REPO = Path("/scratch/gpfs/ZHUANGL/sk7524/PaperLensArXivRelease")
CAP_SHA = "4af324c"   # cap the history at: Add "Where does this signal lie?" section

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({"text.usetex": False, "font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "DejaVu Sans"]})

ROOT = Path(__file__).resolve().parent.parent
SC = ROOT / "logs" / "grid" / "scores"
CACHE = ROOT / "logs" / "grid" / "prepped"
OUT = ROOT / "logs" / "grid" / "maps"; OUT.mkdir(parents=True, exist_ok=True)

MODELS = [
    ("arxivL-7b-text",   "ArXiv-L 7B (text)",   "#1f77b4", "-"),
    ("arxivL-7b-vision", "ArXiv-L 7B (vision)", "#17becf", "-"),
    ("iclr-7b-text",     "ICLR 7B (text)",      "#ff7f0e", "-"),
    ("iclr-7b-vision",   "ICLR 7B (vision)",    "#d62728", "-"),
]

# Venue renders are not git commits — their submission dates are supplied manually.
VENUES = [("colm", "2026-03-31", "COLM submission"),
          ("neurips", "2026-05-05", "NeurIPS submission")]


def load_scores():
    rec = {}
    for m, *_ in MODELS:
        for line in (SC / f"{m}.jsonl").read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("state") == "done":
                    rec[(m, r["paper_id"])] = r["p_accept"]
    return rec


def git_commits():
    """Commits in TRUE git order (oldest->newest), capped at CAP_SHA, restricted to
    those that have a scored cache entry. Returns (pid, 'YYYY-MM-DD', subject)."""
    out = subprocess.run(
        ["git", "-C", str(REPO), "log", "f48b7f8^..HEAD",
         "--format=%H%x1f%cI%x1f%s", "--reverse"],
        capture_output=True, text=True, check=True).stdout
    rows = []
    for line in out.strip().splitlines():
        h, ci, subj = line.split("\x1f", 2)
        rows.append((f"commit_{h[:8]}", ci[:10], subj))
        if h.startswith(CAP_SHA) or "Where does this signal lie" in subj:
            break  # cap (inclusive)
    return rows


def main():
    rec = load_scores()
    scored = {pid for (_, pid) in rec}
    commits = [(pid, date, subj) for pid, date, subj in git_commits() if pid in scored]

    points = VENUES + commits  # full chronological timeline (capped)
    xs = list(range(len(points)))

    def short_date(d):
        try:
            return time.strftime("%m/%d", time.strptime(d, "%Y-%m-%d"))
        except Exception:
            return d
    labels = [f"{short_date(date)}\n({subj[:26]})" for _, date, subj in points]

    fig, ax = plt.subplots(figsize=(max(12, 0.5 * len(points) + 4), 6))
    ax.axhspan(0.5, 1.02, color="#2ca02c", alpha=0.05)
    ax.axhspan(-0.02, 0.5, color="#d62728", alpha=0.05)
    ax.axhline(0.5, color="black", ls="--", lw=1, alpha=0.6)

    for m, disp, color, ls in MODELS:
        ys = [rec.get((m, pid)) for pid, _, _ in points]
        ax.plot(xs, ys, ls, marker="o", ms=4, lw=1.6, color=color, label=disp, alpha=0.9)

    # separators delineating the two venue submissions from the commit history
    ax.axvline(len(VENUES) - 0.5, color="0.6", ls=":", lw=1)
    ax.text(0.5, 1.03, "venue submissions", ha="center", va="bottom", fontsize=8,
            color="0.4", transform=ax.get_xaxis_transform())
    ax.text((len(VENUES) + len(points) - 1) / 2, 1.03, "arXiv-release commit history",
            ha="center", va="bottom", fontsize=8, color="0.4",
            transform=ax.get_xaxis_transform())

    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("p(Accept)", fontsize=13)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=7.5)
    ax.set_title("PaperLens paper — p(Accept) across its lifetime "
                 "(COLM → NeurIPS → arXiv edits)", fontsize=13, pad=18)
    ax.legend(fontsize=9, ncol=2, loc="lower right")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUT / "paperlens_versions.png", dpi=180, bbox_inches="tight")
    plt.close()
    print(f"-> paperlens_versions.png  ({len(points)} points x {len(MODELS)} models)")


if __name__ == "__main__":
    main()
