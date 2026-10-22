"""Three landscape p_accept heatmaps (papers on x-axis, 4x 7B models on y-axis).

  Map 1  golden   : anon_gt golden papers (advisor's favorites)
                    + Decades-Battle grouped with its eccv twin (whitespace gap)
  Map 2  iclr     : anon_ones_we_got_right (unique only) + iclr_best_paper
                    + Transformers-Succinct arxiv/iclr grouped (whitespace gap)
  Map 3  our_lab  : our lab's submissions

Rules:
  - 4 models, 7B only: ArXiv-L text/vision, ICLR text/vision.
  - Papers are UNIQUE across the three maps (MAE/ConvNeXt live in golden, dropped
    from the got-right map).
  - Column label gets a trailing "*" if that paper is in the matching training
    set (arxiv_id in arxiv-large-train, or submission_id in ICLR-train).
  - Version-pairs of the same paper sit as adjacent columns, separated from the
    singletons by a blank spacer column.

Usage:
  python make_advisor_maps.py            # reads logs/grid/{scores,prepped}, writes logs/grid/maps/
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams.update({"text.usetex": False, "font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "DejaVu Sans"]})

ROOT = Path(__file__).resolve().parent.parent
GRID = ROOT / "logs" / "grid"
SCORES = GRID / "scores"
CACHE = GRID / "prepped"
OUT = GRID / "maps"; OUT.mkdir(parents=True, exist_ok=True)
LF = Path("/scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer")

# 4 models, y-axis order
MODELS = [
    ("arxivL-7b-text",   "ArXiv-L 7B (text)"),
    ("arxivL-7b-vision", "ArXiv-L 7B (vision)"),
    ("iclr-7b-text",     "ICLR 7B (text)"),
    ("iclr-7b-vision",   "ICLR 7B (vision)"),
]

# Known arxiv_ids for the named golden papers (subjects have no id in the filename).
# NOTE: the anon_gt Decades-Battle render is a separately-curated PDF that is NOT a
# row in our dataset (its eccv twin 2403.08632 IS) — so it gets no arxiv_id here and
# therefore no asterisk/label, while the eccv render does.
GOLDEN_ARXIV = {
    "anon_gt__ConvNeXt": "2201.03545",
    "anon_gt__DenseNet": "1608.06993",
    "anon_gt__Deconstructing_Diffusion": "2401.14404",
    "anon_gt__Eyes_Wide_Shut": "2401.06209",
    "anon_gt__MAE": "2111.06377",
    "anon_gt__Massive_Activations": "2402.17762",
    "anon_gt__Transformers_without_Normalization": "2503.10622",
    "anon_gt__Wanda": "2306.11695",
}
# Consistent, recognizable labels (the two renders of a paper share one name).
CURATED_LABEL = {
    "anon_gt__ConvNeXt": "ConvNeXt",
    "anon_gt__DenseNet": "DenseNet",
    "anon_gt__Deconstructing_Diffusion": "Deconstructing Diffusion",
    "anon_gt__Eyes_Wide_Shut": "Eyes Wide Shut",
    "anon_gt__MAE": "MAE",
    "anon_gt__Massive_Activations": "Massive Activations",
    "anon_gt__Transformers_without_Normalization": "Transformers w/o Norm",
    "anon_gt__Wanda": "Wanda",
    "anon_gt__Decades_Battle_Dataset_Bias": "Decades Battle Dataset Bias",
    "anon_dataset_bias_eccv__decades_battle_dataset_bias_arxiv_2403.08632": "Decades Battle Dataset Bias",
    "iclr_best_paper__transformers_are_inherently_succinct_arxiv_2510.19315": "Transformers Succinct (arXiv preprint)",
    "iclr_best_paper__transformers_are_inherently_succinct_iclr2026_Yxz92UuPLQ": "Transformers Succinct (ICLR'26 submission)",
}


def load_scores() -> dict:
    rec = {}
    for label, _ in MODELS:
        f = SCORES / f"{label}.jsonl"
        if not f.exists():
            continue
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("state") == "done" and r.get("p_accept") is not None:
                rec[(label, r["paper_id"])] = r["p_accept"]
    return rec


def extract_title(pid: str) -> str:
    """First markdown '# ' header in the cached text row = paper title."""
    f = CACHE / pid / "text.json"
    if not f.exists():
        return pid
    row = json.loads(f.read_text())
    human = next((c["value"] for c in row["conversations"] if c["from"] == "human"), "")
    for ln in human.splitlines():
        s = ln.strip()
        if s.startswith("# "):
            return s[2:].strip()
    return pid


ACRONYMS = {
    "llms": "LLMs", "llm": "LLM", "vlms": "VLMs", "vlm": "VLM", "mllms": "MLLMs",
    "mllm": "MLLM", "gnn": "GNN", "gnns": "GNNs", "vqa": "VQA", "ceo": "CEO",
    "ai": "AI", "rl": "RL", "ml": "ML", "nlp": "NLP", "agentvqa": "AgentVQA",
    "paperlens": "PaperLens", "ceobench": "CEOBench", "occludebench": "OccludeBench",
    "occlude": "Occlude", "nero": "NeRO", "vims": "VLMs",
}


def smart_title(s: str, n: int = 40) -> str:
    out = []
    for tok in re.split(r"(\W+)", s.lower()):
        if tok in ACRONYMS:
            out.append(ACRONYMS[tok])
        elif tok.isalpha():
            out.append(tok.capitalize())
        else:
            out.append(tok)
    t = "".join(out).replace("'S", "'s")
    if len(t) > n:
        t = t[: n - 1].rstrip() + "…"
    return t


def _venue_tag(venue, year):
    """'cvpr',2024 -> 'CVPR'24' ; ('iclr',2025) -> 'ICLR'25'."""
    v = (venue or "").upper() or "?"
    y = f"'{str(year)[2:]}" if year else ""
    return f"{v}{y}"


AX_TRAIN = "data/arxiv_50_50_balanced_per_venue_text_wmetadata_train/data.json"
IC_TRAIN = "data/iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_train/data.json"
AX_TEST = "data/arxiv_50_50_21k_text_wmetadata_filtered24480_y24up_test/data.json"
IC_TEST = "data/iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_test/data.json"
IC_VAL = "data/iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_validation/data.json"


def _collect(files, key, default_venue=None):
    """id -> (answer, venue_tag) from the given data.json files (first wins)."""
    out = {}
    for f in files:
        try:
            d = json.loads((LF / f).read_text())
        except Exception:
            continue
        for r in d:
            m = r.get("_metadata") or {}
            k = m.get(key)
            if not k or k in out:
                continue
            venue = m.get("venue") or default_venue
            year = m.get("conference_year") or m.get("year")
            out[k] = (m.get("answer"), _venue_tag(venue, year))
    return out


def build_label_maps():
    """(arxiv_train_ids, iclr_train_ids, label_lookup). Marks use TRAIN membership only;
    the label line is sourced from train+test+val (so held-out test papers show outcomes)."""
    arxiv_train = set(_collect([AX_TRAIN], "arxiv_id"))
    iclr_train = set(_collect([IC_TRAIN], "submission_id"))
    labels = {}
    labels.update(_collect([AX_TRAIN, AX_TEST], "arxiv_id"))
    labels.update(_collect([IC_TRAIN, IC_TEST, IC_VAL], "submission_id", default_venue="iclr"))
    return arxiv_train, iclr_train, labels


ARXIV_RE = re.compile(r"(\d{4}\.\d{4,5})")
ICLR_RE = re.compile(r"iclr\d{4}_([A-Za-z0-9]{8,})")


def paper_ids(pid: str, subject: str):
    """(arxiv_id, submission_id) for a paper, for train-set membership."""
    aid = GOLDEN_ARXIV.get(pid)
    if not aid:
        m = ARXIV_RE.search(subject)
        aid = m.group(1) if m else None
    m = ICLR_RE.search(subject)
    sid = m.group(1) if m else None
    return aid, sid


# Set in main(): training-membership sets + the broad label lookup.
ARXIV_TRAIN, ICLR_TRAIN, LABELS = set(), set(), {}


def mark_and_label(pid, subject):
    """('*'/'^'/'*^'/'' , '<Accept|Reject> @ VENUE'YY' | None).
    Mark = which TRAIN set(s) the paper is in; label = its ground-truth outcome
    in our dataset (train OR test/val)."""
    aid, sid = paper_ids(pid, subject)
    mark = ("*" if aid in ARXIV_TRAIN else "") + ("^" if sid in ICLR_TRAIN else "")
    lab = LABELS.get(aid) or LABELS.get(sid)
    return mark, (f"{lab[0]} @ {lab[1]}" if lab else None)


def cache_meta():
    meta = {}
    for d in CACHE.iterdir():
        mf = d / "meta.json"
        if d.is_dir() and mf.exists():
            m = json.loads(mf.read_text())
            if m.get("state") == "ok":
                meta[d.name] = m
    return meta


def find(meta, group, contains=None, exclude=None):
    out = []
    for pid, m in meta.items():
        if m.get("group") != group:
            continue
        if contains and contains not in pid:
            continue
        if exclude and any(e in pid for e in exclude):
            continue
        out.append(pid)
    return sorted(out)


def label_for(pid, subject):
    base = CURATED_LABEL.get(pid) or smart_title(extract_title(pid))
    mark, gt = mark_and_label(pid, subject)
    prefix = f"{mark} " if mark else ""
    if gt:
        return f"{prefix}{base}\n({gt})"
    return f"{prefix}{base}"


def bracket(ax, x0, x1, label):
    """Square bracket above the heatmap spanning columns [x0, x1] with a label."""
    yb, tip = -0.85, -0.68            # origin='upper': smaller y = above the grid
    ax.plot([x0, x0, x1, x1], [tip, yb, yb, tip], color="0.25", lw=1.1, clip_on=False)
    ax.text((x0 + x1) / 2, yb - 0.12, label, ha="center", va="bottom",
            fontsize=10, fontweight="bold", color="0.15", clip_on=False)


def render(title, cols, rec, meta, fname, brackets=None):
    """cols: flat list of paper_ids with None = whitespace spacer (no tick/label).
    brackets: optional list of (col_i0, col_i1, label) -> square bracket above."""
    labels, ticks, M = [], [], np.full((len(MODELS), len(cols)), np.nan)
    for j, pid in enumerate(cols):
        if pid is None:
            continue                  # spacer: no tick, no label
        subj = meta[pid].get("subject", pid)
        ticks.append(j); labels.append(label_for(pid, subj))
        for i, (ml, _) in enumerate(MODELS):
            v = rec.get((ml, pid))
            if v is not None:
                M[i, j] = v

    fig_w = max(8, 0.95 * len(cols) + 3)
    fig, ax = plt.subplots(figsize=(fig_w, 4.2))
    im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels([d for _, d in MODELS], fontsize=10)
    for i in range(len(MODELS)):
        for j in range(len(cols)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="black")
    for (a, b, lab) in brackets or []:
        bracket(ax, a, b, lab)
    ax.set_title(title, fontsize=13, pad=28 if brackets else 10)
    cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cb.set_label("p(Accept)", fontsize=9)
    fig.text(0.01, 0.01, "* = in ArXiv-L train, ^ = in ICLR train, *^ = both;  "
             "(label @ VENUE'YY) = ground-truth outcome in our dataset",
             fontsize=7.5, color="0.35")
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(OUT / fname, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"-> {fname}  ({sum(1 for c in cols if c)} papers x {len(MODELS)} models)")


def main():
    global ARXIV_TRAIN, ICLR_TRAIN, LABELS
    rec = load_scores()
    meta = cache_meta()
    ARXIV_TRAIN, ICLR_TRAIN, LABELS = build_label_maps()

    # ---- Map 1: golden (advisor's favorites) ----
    # Decades-Battle shown as a side-by-side pair (anon_gt render + eccv render); only
    # the eccv render is in our dataset, so only it carries a mark/label.
    golden_single = sorted(p for p in find(meta, "anon_gt") if p != "anon_gt__Decades_Battle_Dataset_Bias")
    decades_pair = ["anon_gt__Decades_Battle_Dataset_Bias"] + find(meta, "anon_dataset_bias_eccv")
    render("Golden papers (advisor's favorites) — PaperLens p(Accept)",
           golden_single + [None] + decades_pair, rec, meta, "map1_golden.png")

    # ---- Map 2: ICLR Best Papers | Papers We Got Right (bracketed, white-block sep) ----
    # The two Transformers-Succinct renders are split by a whitespace column.
    llms = find(meta, "iclr_best_paper", contains="llms_get_lost")
    sa = find(meta, "iclr_best_paper", contains="succinct_arxiv")
    si = find(meta, "iclr_best_paper", contains="succinct_iclr")
    got = sorted(find(meta, "anon_ones_we_got_right", exclude=["2111.06377", "2201.03545"]))
    # two Transformers-Succinct renders kept ADJACENT as a pair, the pair set off by
    # whitespace from llms (and from the got-right block).
    best_block = llms + [None] + sa + si
    cols = best_block + [None] + got
    gstart = len(best_block) + 1
    render("ICLR papers — PaperLens p(Accept)", cols, rec, meta, "map2_iclr.png",
           brackets=[(0, len(best_block) - 1, "ICLR Best Papers"),
                     (gstart, gstart + len(got) - 1, "Papers We Got Right")])

    # ---- Map 3: our lab (PaperLens excluded — it's our own model) ----
    our_lab = [p for p in find(meta, "our_lab") if "PaperLens" not in p]
    render("Our lab's submissions — PaperLens p(Accept)", our_lab, rec, meta, "map3_ourlab.png")


if __name__ == "__main__":
    main()
