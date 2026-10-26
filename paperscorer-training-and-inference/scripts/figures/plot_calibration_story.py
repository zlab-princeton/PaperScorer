#!/usr/bin/env python3
"""
Two figures showing the calibration / cross-distribution / bias story:

  fig_iclr_crosseval.pdf : 3-panel — raw ACC, calibrated ACC (Δ-colored), AUC.
                           Each cell shows [25 / 26] year-split numbers.
  fig_calibration_bias.pdf : 2-panel — arxiv per-venue dumbbells (raw→calib),
                             pred-A% vs gold-A% bias scatter.

Source data: ICLR + arxiv y24up text cross-eval at ckpt 1322.
Computation logic mirrors scripts/iclr_arxiv_calibrated_eval.py.
"""
from __future__ import annotations
import pathlib

import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
})
LSIZE, TSIZE, TICK = 18, 18, 14

ROOT = pathlib.Path(__file__).resolve().parents[2]
RES  = ROOT / "results/final_sweep_v7_datasweepv3/optim_search_2026"
DATA = ROOT / "data"
OUT  = ROOT / "tmp_latex_dir/figures"
OUT.mkdir(parents=True, exist_ok=True)
CKPT = 1322

# ---------- ICLR config ----------
ICLR_TEST = {
    "50_50": DATA / "iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_test/data.json",
    "40_60": DATA / "iclr_2020_2023_2025_2026_40_60_original_text_v7_filtered_test/data.json",
    "30_70": DATA / "iclr_2020_2023_2025_2026_30_70_original_text_v7_filtered_test/data.json",
}
ICLR_VAL = {
    "50_50": DATA / "iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_validation/data.json",
    "40_60": DATA / "iclr_2020_2023_2025_2026_40_60_original_text_v7_filtered_validation_DERIVED/data.json",
    "30_70": DATA / "iclr_2020_2023_2025_2026_30_70_original_text_v7_filtered_validation_DERIVED/data.json",
}
ICLR_SHORTS = {"50_50": "bz32_lr1e-6_text",
               "40_60": "bz32_lr1e-6_text_40_60",
               "30_70": "bz32_lr1e-6_text_30_70"}
RATIOS = ["50_50", "40_60", "30_70"]
LBL = {"50_50": "50/50", "40_60": "40/60", "30_70": "30/70"}

ARXIV_RES = ROOT / "results/cross_conference_arxiv_y24up"
ARXIV_TEST_PATH = DATA / "arxiv_50_50_21k_text_wmetadata_filtered24480_y24up_test/data.json"
ARXIV_VAL_PATH  = DATA / "arxiv_50_50_21k_text_wmetadata_filtered24480_y24up_validation/data.json"
ACL_FAMILY = {"acl", "emnlp", "naacl"}

RATIO_COLOR = {"50_50": "#6098FF", "40_60": "#FECC81", "30_70": "#7FB069"}

# ---------- helpers ----------
def extract_pred(t):
    s = (t or "").lower()
    if "\\boxed{accept}" in s or "boxed{accept}" in s: return "accept"
    if "\\boxed{reject}" in s or "boxed{reject}" in s: return "reject"
    if "accept" in s: return "accept"
    if "reject" in s: return "reject"
    return None

def score_logodds(row):
    p = row.get("predict","")
    chosen = 1 if "Accept" in p else (0 if "Reject" in p else None)
    if chosen is None: return None
    la = row.get("logprob_accept") or []
    lr = row.get("logprob_reject") or []
    tl = row.get("token_logprobs") or []
    for k in range(min(len(tl), len(la), len(lr))):
        if la[k] is not None and lr[k] is not None and tl[k] is not None:
            if math.isclose(tl[k], max(la[k], lr[k]), abs_tol=1e-3):
                return la[k] - lr[k]
    return 6.0 if chosen == 1 else -6.0

def auc(scores, labels):
    if not scores or len(set(labels)) < 2: return None
    pairs = sorted(zip(scores, labels))
    n = len(pairs); npos = sum(labels); nneg = n - npos
    rs = sum(i+1 for i,(s,l) in enumerate(pairs) if l == 1)
    return (rs - npos*(npos+1)/2) / (npos*nneg)

def best_tau(pairs):
    if not pairs: return 0.0
    pairs = sorted(pairs)
    n = len(pairs); npos = sum(g for _,g in pairs)
    correct = npos
    best_acc, best_tau = correct/n, -math.inf
    for s,g in pairs:
        correct += (1 if g == 0 else -1)
        if correct/n > best_acc:
            best_acc, best_tau = correct/n, s
    return best_tau

def load_pairs(jsonl_path, meta_path, year_min=None, with_meta=False):
    if not jsonl_path.exists() or not meta_path.exists(): return []
    meta = json.loads(meta_path.read_text())
    out = []
    with jsonl_path.open() as f:
        for i, line in enumerate(f):
            if i >= len(meta): break
            yr = (meta[i].get("_metadata") or {}).get("year")
            try: yr = int(yr) if yr is not None else None
            except: pass
            if year_min is not None and (yr is None or yr < year_min): continue
            r = json.loads(line)
            g = extract_pred(r.get("label",""))
            s = score_logodds(r)
            if g is None or s is None: continue
            gold = 1 if g == "accept" else 0
            if with_meta:
                m = meta[i].get("_metadata") or {}
                v = (m.get("pl_venue") or m.get("venue") or "?").lower()
                if v in ACL_FAMILY: v = "acl_family"
                out.append((yr, gold, s, v))
            else:
                out.append((yr, gold, s))
    return out

# ---------- ICLR cell metrics (matches eval script) ----------
def iclr_test_path(train, test):
    short = ICLR_SHORTS[train]
    if train == "50_50":
        if test == "50_50":
            return RES / short / f"finetuned-ckpt-{CKPT}.jsonl"
        return RES / "ratio_crossval_clean" / short / f"test_{test}" / f"finetuned-ckpt-{CKPT}.jsonl"
    tag = "balanced" if test == "50_50" else test
    return RES / "ratio_sweep" / short / tag / f"finetuned-ckpt-{CKPT}.jsonl"

def iclr_val_path(train, val):
    short = ICLR_SHORTS[train]
    if train == "50_50" and val == "50_50":
        return RES / short / f"validation-ckpt-{CKPT}.jsonl"
    val_tag = "balanced" if val == "50_50" else val
    return ROOT / "results/iclr_val_calib" / short / f"val_{val_tag}" / f"finetuned-ckpt-{CKPT}.jsonl"

def iclr_cell(train, test):
    test_pairs = load_pairs(iclr_test_path(train, test), ICLR_TEST[test], year_min=2025)
    val_pairs  = load_pairs(iclr_val_path(train, test),  ICLR_VAL[test],  year_min=2025)
    if not val_pairs:
        val_pairs = load_pairs(iclr_val_path(train, test), ICLR_VAL[test], year_min=None)
    tau = best_tau([(s,g) for _,g,s in val_pairs])
    out = {"tau": tau}
    for yr in (2025, 2026):
        sub = [(g,s) for y,g,s in test_pairs if y == yr]
        n = len(sub)
        if n == 0:
            out[yr] = dict(acc_raw=None, acc_cal=None, auc=None, n=0,
                           pred_a=None, gold_a=None)
            continue
        acc_raw = sum(1 for g,s in sub if (s>0)==(g==1))/n*100
        acc_cal = sum(1 for g,s in sub if (s>tau)==(g==1))/n*100
        scores = [s for g,s in sub]; gold = [g for g,s in sub]
        au = auc(scores, gold)
        pred_a = sum(1 for g,s in sub if s>0)/n*100  # raw predicted accept rate
        gold_a = sum(g for g,s in sub)/n*100
        out[yr] = dict(acc_raw=acc_raw, acc_cal=acc_cal, auc=au, n=n,
                       pred_a=pred_a, gold_a=gold_a)
    return out


# =======================================================
# Build arxiv per-(train, venue) test + val data
# =======================================================
def build_arxiv_data():
    out = {}
    for ratio in RATIOS:
        short = ICLR_SHORTS[ratio]
        val_pairs  = load_pairs(ARXIV_RES / short / "arxiv_val"  / f"finetuned-ckpt-{CKPT}.jsonl",
                                ARXIV_VAL_PATH, with_meta=True)
        test_pairs = load_pairs(ARXIV_RES / short / "arxiv_eval" / f"finetuned-ckpt-{CKPT}.jsonl",
                                ARXIV_TEST_PATH, with_meta=True)
        val_by_v  = defaultdict(list); test_by_v = defaultdict(list)
        for _,g,s,v in val_pairs:  val_by_v[v].append((s,g))
        for _,g,s,v in test_pairs: test_by_v[v].append((s,g))
        global_tau = best_tau([(s,g) for _,g,s,_ in val_pairs])
        thr = {}
        for v, pairs in val_by_v.items():
            thr[v] = (best_tau(pairs), False) if len(pairs) >= 10 else (global_tau, True)
        for v in test_by_v:
            thr.setdefault(v, (global_tau, True))
        out[ratio] = {"val_by_v": val_by_v, "test_by_v": test_by_v, "thr": thr,
                       "global_tau": global_tau}
    return out


# =======================================================
# FIGURE 1: 2 rows (ICLR + Arxiv) × 3 cols (Raw, Calib, AUC)
# =======================================================
def fig1_test_results():
    cells = {(t, e): iclr_cell(t, e) for t in RATIOS for e in RATIOS}
    arxiv = build_arxiv_data()

    # ---------- ICLR matrices ----------
    iclr_raw = np.zeros((3, 3)); iclr_cal = np.zeros((3, 3)); iclr_auc = np.zeros((3, 3))
    for i, t in enumerate(RATIOS):
        for j, e in enumerate(RATIOS):
            c = cells[(t, e)]
            iclr_raw[i, j] = (c[2025]["acc_raw"] + c[2026]["acc_raw"]) / 2
            iclr_cal[i, j] = (c[2025]["acc_cal"] + c[2026]["acc_cal"]) / 2
            iclr_auc[i, j] = (c[2025]["auc"] + c[2026]["auc"]) / 2
    iclr_delta = iclr_cal - iclr_raw

    # ---------- Arxiv matrices ----------
    venues = sorted({v for r in RATIOS for v in arxiv[r]["test_by_v"]},
                    key=lambda v: -sum(len(arxiv[r]["test_by_v"].get(v, [])) for r in RATIOS))
    nv = len(venues)
    ax_raw = np.full((nv, 3), np.nan)
    ax_cal = np.full((nv, 3), np.nan)
    ax_auc = np.full((nv, 3), np.nan)
    ax_n   = np.zeros((nv, 3), dtype=int)
    for j, ratio in enumerate(RATIOS):
        for i, v in enumerate(venues):
            tp = arxiv[ratio]["test_by_v"].get(v, [])
            if not tp: continue
            n = len(tp); tau, _ = arxiv[ratio]["thr"][v]
            ax_raw[i, j] = sum(1 for s,g in tp if (s>0)==(g==1))/n*100
            ax_cal[i, j] = sum(1 for s,g in tp if (s>tau)==(g==1))/n*100
            scores = [s for s,_ in tp]; gold = [g for _,g in tp]
            au = auc(scores, gold)
            ax_auc[i, j] = au if au is not None else np.nan
            ax_n[i, j] = n
    ax_delta = ax_cal - ax_raw

    # ---------- Arxiv OVERALL (rolled up across all venues) ----------
    ovr_raw = np.zeros((1, 3)); ovr_cal = np.zeros((1, 3)); ovr_auc = np.zeros((1, 3))
    ovr_n = np.zeros((1, 3), dtype=int)
    for j, ratio in enumerate(RATIOS):
        all_pairs = []
        n_calib_correct = 0
        for v, tp in arxiv[ratio]["test_by_v"].items():
            tau, _ = arxiv[ratio]["thr"][v]
            n_calib_correct += sum(1 for s,g in tp if (s>tau)==(g==1))
            all_pairs.extend(tp)
        n = len(all_pairs)
        if n == 0: continue
        ovr_raw[0, j] = sum(1 for s,g in all_pairs if (s>0)==(g==1)) / n * 100
        ovr_cal[0, j] = n_calib_correct / n * 100
        scores = [s for s,_ in all_pairs]; gold = [g for _,g in all_pairs]
        ovr_auc[0, j] = auc(scores, gold)
        ovr_n[0, j] = n

    # ---------- Figure layout (3 rows, single viridis colormap) ----------
    fig = plt.figure(figsize=(20, 18))
    gs = fig.add_gridspec(3, 3,
                          height_ratios=[1.0, 2.0, 0.35],  # iclr / arxiv-perven / overall
                          hspace=0.32, wspace=0.32)

    def draw_heatmap(ax, mat, vmin, vmax, title, cbar_label,
                     row_labels=None, col_labels=None,
                     row_axis_label=None, col_axis_label=None,
                     cell_text_fn=None, txt_fontsize=TICK):
        im = ax.imshow(mat, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
        if col_labels is not None:
            ax.set_xticks(range(len(col_labels)))
            ax.set_xticklabels(col_labels, fontsize=TICK)
        if row_labels is not None:
            ax.set_yticks(range(len(row_labels)))
            ax.set_yticklabels(row_labels, fontsize=TICK - 2)
        if col_axis_label: ax.set_xlabel(col_axis_label, fontsize=LSIZE - 2)
        if row_axis_label: ax.set_ylabel(row_axis_label, fontsize=LSIZE - 2)
        ax.set_title(title, fontsize=TSIZE - 2, pad=10)
        if cell_text_fn is not None:
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    txt = cell_text_fn(i, j)
                    if txt is None: continue
                    # high contrast text: white if cell is dark
                    val = mat[i, j]
                    if not np.isnan(val):
                        norm = (val - vmin) / max(vmax - vmin, 1e-9)
                        color = "white" if norm < 0.5 else "black"
                    else:
                        color = "black"
                    ax.text(j, i, txt, ha="center", va="center",
                            fontsize=txt_fontsize, color=color, linespacing=1.0)
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(cbar_label, fontsize=TICK - 1)
        cb.ax.tick_params(labelsize=TICK - 2)

    # ===== Row 1: ICLR 3x3 (TRAIN ratio on x, TEST ratio on y, matching arxiv rows) =====
    # Transpose: rows now = test ratio, cols = train ratio.
    iclr_raw = iclr_raw.T
    iclr_cal = iclr_cal.T
    iclr_auc = iclr_auc.T

    R = [LBL[r] for r in RATIOS]
    iclr_all_acc = np.concatenate([iclr_raw.flatten(), iclr_cal.flatten()])
    acc_vmin = iclr_all_acc.min() - 1
    acc_vmax = iclr_all_acc.max() + 1

    # NB: i = test ratio idx (row), j = train ratio idx (col)
    def cell_t(i, j): return RATIOS[j], RATIOS[i]   # (train, test) tuple key
    ax = fig.add_subplot(gs[0, 0])
    draw_heatmap(ax, iclr_raw, acc_vmin, acc_vmax,
                 "ICLR Raw ACC", "ACC (%)",
                 R, R, "Test ratio", "Train ratio",
                 lambda i, j: f"{cells[cell_t(i,j)][2025]['acc_raw']:.1f}\n"
                              f"{cells[cell_t(i,j)][2026]['acc_raw']:.1f}",
                 txt_fontsize=TICK - 1)
    def iclr_cal_text(i, j):
        c = cells[cell_t(i, j)]
        d25 = c[2025]['acc_cal'] - c[2025]['acc_raw']
        d26 = c[2026]['acc_cal'] - c[2026]['acc_raw']
        return (f"{c[2025]['acc_cal']:.1f}({d25:+.1f})\n"
                f"{c[2026]['acc_cal']:.1f}({d26:+.1f})")
    ax = fig.add_subplot(gs[0, 1])
    draw_heatmap(ax, iclr_cal, acc_vmin, acc_vmax,
                 "ICLR Calibrated ACC  [calib(Δ)]", "ACC (%)",
                 R, R, "Test ratio", "Train ratio",
                 iclr_cal_text, txt_fontsize=TICK - 3)
    auc_vmin = iclr_auc.min() - 0.01; auc_vmax = max(iclr_auc.max(), ax_auc[~np.isnan(ax_auc)].max()) + 0.01 if (~np.isnan(ax_auc)).any() else iclr_auc.max() + 0.01
    ax = fig.add_subplot(gs[0, 2])
    draw_heatmap(ax, iclr_auc, auc_vmin, auc_vmax,
                 "ICLR ROC-AUC", "AUC", R, R, "Test ratio", "Train ratio",
                 lambda i, j: f"{cells[cell_t(i,j)][2025]['auc']:.3f}\n"
                              f"{cells[cell_t(i,j)][2026]['auc']:.3f}",
                 txt_fontsize=TICK - 1)

    # ===== Row 2: Arxiv per-venue (single number per panel, single viridis) =====
    valid_acc = np.concatenate([ax_raw[~np.isnan(ax_raw)], ax_cal[~np.isnan(ax_cal)]])
    arx_acc_vmin = valid_acc.min() - 1; arx_acc_vmax = valid_acc.max() + 1

    ax = fig.add_subplot(gs[1, 0])
    draw_heatmap(ax, ax_raw, arx_acc_vmin, arx_acc_vmax,
                 "Arxiv Raw ACC (per venue)", "ACC (%)",
                 venues, R, None, "Train ratio",
                 lambda i, j: (f"{ax_raw[i,j]:.0f}" if not np.isnan(ax_raw[i,j]) else None),
                 txt_fontsize=TICK - 2)
    def arx_cal_text(i, j):
        if np.isnan(ax_cal[i, j]): return None
        d = ax_cal[i, j] - ax_raw[i, j]
        return f"{ax_cal[i,j]:.0f}({d:+.0f})"
    ax = fig.add_subplot(gs[1, 1])
    draw_heatmap(ax, ax_cal, arx_acc_vmin, arx_acc_vmax,
                 "Arxiv Calibrated ACC (per venue)  [calib(Δ)]", "ACC (%)",
                 venues, R, None, "Train ratio",
                 arx_cal_text, txt_fontsize=TICK - 3)
    valid_auc = ax_auc[~np.isnan(ax_auc)]
    arx_auc_vmin = valid_auc.min() - 0.01; arx_auc_vmax = valid_auc.max() + 0.01
    ax = fig.add_subplot(gs[1, 2])
    draw_heatmap(ax, ax_auc, arx_auc_vmin, arx_auc_vmax,
                 "Arxiv ROC-AUC (per venue)", "AUC",
                 venues, R, None, "Train ratio",
                 lambda i, j: (f"{ax_auc[i,j]:.3f}\n(n={ax_n[i,j]})"
                               if not np.isnan(ax_auc[i,j]) else None),
                 txt_fontsize=TICK - 4)

    # ===== Row 3: Arxiv OVERALL (rolled up across all venues) =====
    ax = fig.add_subplot(gs[2, 0])
    draw_heatmap(ax, ovr_raw, arx_acc_vmin, arx_acc_vmax,
                 "Arxiv Raw ACC (overall)", "ACC (%)",
                 ["overall"], R, None, "Train ratio",
                 lambda i, j: f"{ovr_raw[0,j]:.1f}\n(n={ovr_n[0,j]})",
                 txt_fontsize=TICK - 1)
    ax = fig.add_subplot(gs[2, 1])
    draw_heatmap(ax, ovr_cal, arx_acc_vmin, arx_acc_vmax,
                 "Arxiv Calib ACC (overall)  [calib(Δ)]", "ACC (%)",
                 ["overall"], R, None, "Train ratio",
                 lambda i, j: f"{ovr_cal[0,j]:.1f}({ovr_cal[0,j]-ovr_raw[0,j]:+.1f})\n(n={ovr_n[0,j]})",
                 txt_fontsize=TICK - 1)
    ax = fig.add_subplot(gs[2, 2])
    draw_heatmap(ax, ovr_auc, arx_auc_vmin, arx_auc_vmax,
                 "Arxiv ROC-AUC (overall)", "AUC",
                 ["overall"], R, None, "Train ratio",
                 lambda i, j: f"{ovr_auc[0,j]:.3f}\n(n={ovr_n[0,j]})",
                 txt_fontsize=TICK - 1)

    fig.suptitle(
        "Cross-eval test results — ICLR (top, [25/26] per cell), "
        "Arxiv y24up per-venue (middle), Arxiv overall (bottom)",
        fontsize=TSIZE + 1, y=0.995, fontweight="bold")
    out_pdf = OUT / "fig_iclr_crosseval.pdf"
    out_png = OUT / "fig_iclr_crosseval.png"
    plt.savefig(out_pdf, dpi=200, bbox_inches="tight", pad_inches=0.2)
    plt.savefig(out_png, dpi=150, bbox_inches="tight", pad_inches=0.2)
    plt.close()
    print(f"saved {out_pdf}, {out_png}")
    return cells, arxiv, venues, ax_n


# =======================================================
# FIGURE 1.5 (NEW): Validation set we calibrated against
# =======================================================
def fig_validation_summary(arxiv, venues, ax_n_test):
    """Show what we calibrated against:
       - ICLR val acc at τ=0 vs τ* (3x3 heatmaps), plus τ* itself
       - Arxiv val acc per-venue at τ* (or per-venue τ* itself)
    """
    fig = plt.figure(figsize=(20, 8.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.5], wspace=0.32)

    # --- ICLR val raw vs calibrated acc ---
    iclr_val_raw = np.zeros((3, 3))
    iclr_val_cal = np.zeros((3, 3))
    iclr_tau     = np.zeros((3, 3))
    for i, t in enumerate(RATIOS):
        for j, vr in enumerate(RATIOS):
            val_pairs = load_pairs(iclr_val_path(t, vr), ICLR_VAL[vr], year_min=2025)
            if not val_pairs:
                val_pairs = load_pairs(iclr_val_path(t, vr), ICLR_VAL[vr], year_min=None)
            tau = best_tau([(s,g) for _,g,s in val_pairs])
            n = len(val_pairs)
            if n == 0:
                iclr_val_raw[i, j] = np.nan; iclr_val_cal[i, j] = np.nan; iclr_tau[i, j] = np.nan
                continue
            iclr_val_raw[i, j] = sum(1 for _,g,s in val_pairs if (s>0)==(g==1))/n*100
            iclr_val_cal[i, j] = sum(1 for _,g,s in val_pairs if (s>tau)==(g==1))/n*100
            iclr_tau[i, j]     = tau
    iclr_val_delta = iclr_val_cal - iclr_val_raw

    # Panel 1: ICLR val ACC (raw → calib via τ*); color = Δ
    ax1 = fig.add_subplot(gs[0, 0])
    valid_d = iclr_val_delta[~np.isnan(iclr_val_delta)]
    dmax = max(abs(valid_d.min()), abs(valid_d.max())) if valid_d.size else 1
    im = ax1.imshow(iclr_val_delta, cmap="RdBu_r", vmin=-dmax, vmax=dmax, aspect="auto")
    ax1.set_xticks(range(3)); ax1.set_xticklabels([LBL[r] for r in RATIOS], fontsize=TICK)
    ax1.set_yticks(range(3)); ax1.set_yticklabels([LBL[r] for r in RATIOS], fontsize=TICK)
    ax1.set_xlabel("Val ratio", fontsize=LSIZE - 2)
    ax1.set_ylabel("Train ratio", fontsize=LSIZE - 2)
    ax1.set_title("ICLR validation ACC (raw → calib via τ*)\nyear ≥ 2025",
                  fontsize=TSIZE - 2, pad=10)
    for i in range(3):
        for j in range(3):
            if np.isnan(iclr_val_raw[i, j]): continue
            txt = f"{iclr_val_raw[i, j]:.1f}→{iclr_val_cal[i, j]:.1f}"
            ax1.text(j, i, txt, ha="center", va="center",
                     fontsize=TICK - 2, color="black")
    cb = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cb.set_label("Δ ACC (calib − raw, on val)", fontsize=TICK - 1)
    cb.ax.tick_params(labelsize=TICK - 2)

    # Panel 2: τ* matrix for ICLR
    ax2 = fig.add_subplot(gs[0, 1])
    valid_tau = iclr_tau[~np.isnan(iclr_tau)]
    tmax = max(abs(valid_tau.min()), abs(valid_tau.max())) if valid_tau.size else 1
    im = ax2.imshow(iclr_tau, cmap="PuOr_r", vmin=-tmax, vmax=tmax, aspect="auto")
    ax2.set_xticks(range(3)); ax2.set_xticklabels([LBL[r] for r in RATIOS], fontsize=TICK)
    ax2.set_yticks(range(3)); ax2.set_yticklabels([LBL[r] for r in RATIOS], fontsize=TICK)
    ax2.set_xlabel("Val ratio", fontsize=LSIZE - 2)
    ax2.set_ylabel("Train ratio", fontsize=LSIZE - 2)
    ax2.set_title("ICLR τ* threshold from val", fontsize=TSIZE - 2, pad=10)
    for i in range(3):
        for j in range(3):
            if np.isnan(iclr_tau[i, j]): continue
            t_str = f"{iclr_tau[i, j]:+.2f}" if abs(iclr_tau[i, j]) < 5 else f"{iclr_tau[i, j]:+.1f}"
            ax2.text(j, i, t_str, ha="center", va="center",
                     fontsize=TICK, color="black")
    cb = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cb.set_label("τ* (log-odds threshold)", fontsize=TICK - 1)
    cb.ax.tick_params(labelsize=TICK - 2)

    # Panel 3: arxiv val per-venue τ* (or val acc)
    ax3 = fig.add_subplot(gs[0, 2])
    nv = len(venues)
    arx_tau = np.full((nv, 3), np.nan)
    arx_val_acc = np.full((nv, 3), np.nan)
    arx_val_n = np.zeros((nv, 3), dtype=int)
    for j, ratio in enumerate(RATIOS):
        for i, v in enumerate(venues):
            vp = arxiv[ratio]["val_by_v"].get(v, [])
            tau, _is_global = arxiv[ratio]["thr"].get(v, (arxiv[ratio]["global_tau"], True))
            arx_tau[i, j] = tau
            n = len(vp)
            if n > 0:
                arx_val_acc[i, j] = sum(1 for s,g in vp if (s>tau)==(g==1))/n*100
                arx_val_n[i, j] = n
    valid_tau = arx_tau[~np.isnan(arx_tau)]
    tmax = max(abs(valid_tau.min()), abs(valid_tau.max())) if valid_tau.size else 1
    im = ax3.imshow(arx_tau, cmap="PuOr_r", vmin=-tmax, vmax=tmax, aspect="auto")
    ax3.set_xticks(range(3)); ax3.set_xticklabels([LBL[r] for r in RATIOS], fontsize=TICK)
    ax3.set_yticks(range(nv)); ax3.set_yticklabels(venues, fontsize=TICK - 2)
    ax3.set_xlabel("Train ratio", fontsize=LSIZE - 2)
    ax3.set_title("Arxiv per-venue τ* from val\n(val acc / n)", fontsize=TSIZE - 2, pad=10)
    for i in range(nv):
        for j in range(3):
            if np.isnan(arx_tau[i, j]): continue
            tau, is_g = arxiv[RATIOS[j]]["thr"].get(venues[i], (arxiv[RATIOS[j]]["global_tau"], True))
            tau_s = f"{tau:+.1f}" if abs(tau) >= 1 else f"{tau:+.2f}"
            star = "g" if is_g else ""
            n = arx_val_n[i, j]
            ax3.text(j, i, f"{tau_s}{star}\n({arx_val_acc[i,j]:.0f},n={n})",
                     ha="center", va="center", fontsize=TICK - 4, color="black",
                     linespacing=0.95)
    cb = plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    cb.set_label("τ* (log-odds threshold)", fontsize=TICK - 1)
    cb.ax.tick_params(labelsize=TICK - 2)

    fig.suptitle("Validation set: where the calibration thresholds τ* came from",
                 fontsize=TSIZE + 1, y=1.02, fontweight="bold")
    out_pdf = OUT / "fig_validation_summary.pdf"
    out_png = OUT / "fig_validation_summary.png"
    plt.savefig(out_pdf, dpi=200, bbox_inches="tight", pad_inches=0.2)
    plt.savefig(out_png, dpi=150, bbox_inches="tight", pad_inches=0.2)
    plt.close()
    print(f"saved {out_pdf}, {out_png}")


# =======================================================
# FIGURE 2: Arxiv calibration dumbbells | ICLR bias scatter
# =======================================================
def fig2_calibration_and_bias(iclr_cells):
    fig = plt.figure(figsize=(20, 7.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1.0], wspace=0.28)

    # ---- Panel 1: Arxiv per-venue dumbbells ----
    ax1 = fig.add_subplot(gs[0, 0])

    arxiv_data = {}
    for ratio in RATIOS:
        short = ICLR_SHORTS[ratio]
        val_pairs  = load_pairs(ARXIV_RES / short / "arxiv_val"  / f"finetuned-ckpt-{CKPT}.jsonl",
                                ARXIV_VAL_PATH, with_meta=True)
        test_pairs = load_pairs(ARXIV_RES / short / "arxiv_eval" / f"finetuned-ckpt-{CKPT}.jsonl",
                                ARXIV_TEST_PATH, with_meta=True)
        val_by_v  = defaultdict(list)
        test_by_v = defaultdict(list)
        for _,g,s,v in val_pairs:  val_by_v[v].append((s,g))
        for _,g,s,v in test_pairs: test_by_v[v].append((s,g))
        global_tau = best_tau([(s,g) for _,g,s,_ in val_pairs])
        thr = {}
        for v, pairs in val_by_v.items():
            thr[v] = best_tau(pairs) if len(pairs) >= 10 else global_tau
        for v in test_by_v:
            thr.setdefault(v, global_tau)
        arxiv_data[ratio] = {"test_by_v": test_by_v, "thr": thr}

    # Sort venues by total test count (desc) so big ones are at top
    venues = sorted({v for r in RATIOS for v in arxiv_data[r]["test_by_v"]},
                    key=lambda v: -sum(len(arxiv_data[r]["test_by_v"].get(v, [])) for r in RATIOS))

    n_v = len(venues)
    y_positions = np.arange(n_v)
    bar_offsets = {"50_50": -0.25, "40_60": 0.0, "30_70": +0.25}

    for ratio in RATIOS:
        d = arxiv_data[ratio]
        col = RATIO_COLOR[ratio]
        offs = bar_offsets[ratio]
        for i, v in enumerate(venues):
            tp = d["test_by_v"].get(v, [])
            if not tp: continue
            n = len(tp)
            tau = d["thr"][v]
            raw = sum(1 for s,g in tp if (s>0)==(g==1)) / n * 100
            cal = sum(1 for s,g in tp if (s>tau)==(g==1)) / n * 100
            y = i + offs
            ax1.plot([raw, cal], [y, y], color=col, linewidth=2.5, alpha=0.55, zorder=2)
            ax1.scatter([raw], [y], color=col, marker="o", s=70, zorder=3,
                        edgecolor="black", linewidth=0.5,
                        label=f"{LBL[ratio]} raw" if i == 0 else None)
            ax1.scatter([cal], [y], color=col, marker="D", s=70, zorder=4,
                        edgecolor="black", linewidth=0.5,
                        label=f"{LBL[ratio]} calib" if i == 0 else None)

    ax1.set_yticks(range(n_v)); ax1.set_yticklabels(venues, fontsize=TICK)
    ax1.invert_yaxis()
    ax1.set_xlabel("Accuracy (%)", fontsize=LSIZE - 2)
    ax1.set_xlim(20, 105)
    ax1.axvline(x=50, color="gray", linestyle=":", alpha=0.4, linewidth=1)
    ax1.set_title("Arxiv per-venue: raw → calibrated accuracy",
                  fontsize=TSIZE - 2, pad=10)
    ax1.grid(True, axis="x", linestyle=":", alpha=0.4)
    ax1.legend(fontsize=TICK - 1, loc="lower right", ncol=3,
               framealpha=0.95, columnspacing=1.0)

    # ---- Panel 2: Bias scatter (pred-A% vs gold-A%) ----
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot([0, 100], [0, 100], "--", color="gray", alpha=0.6, linewidth=1, label="no bias (y=x)")

    for t in RATIOS:
        for e in RATIOS:
            c = iclr_cells[(t, e)]
            for yr, marker in [(2025, "o"), (2026, "s")]:
                row = c[yr]
                if row["pred_a"] is None: continue
                ax2.scatter([row["gold_a"]], [row["pred_a"]],
                            color=RATIO_COLOR[t], marker=marker, s=110,
                            edgecolor="black", linewidth=0.6, zorder=3)

    # Legend (per train ratio + year markers)
    for t in RATIOS:
        ax2.scatter([], [], color=RATIO_COLOR[t], s=110, edgecolor="black",
                    linewidth=0.6, label=f"Train {LBL[t]}")
    ax2.scatter([], [], color="gray", marker="o", s=110, edgecolor="black",
                linewidth=0.6, label="2025")
    ax2.scatter([], [], color="gray", marker="s", s=110, edgecolor="black",
                linewidth=0.6, label="2026")

    ax2.set_xlabel("Gold accept rate (%)", fontsize=LSIZE - 2)
    ax2.set_ylabel("Predicted accept rate (%, raw τ=0)", fontsize=LSIZE - 2)
    ax2.set_xlim(20, 80); ax2.set_ylim(0, 100)
    ax2.set_title("ICLR bias direction (raw decoding)",
                  fontsize=TSIZE - 2, pad=10)
    ax2.grid(True, linestyle=":", alpha=0.4)
    ax2.legend(fontsize=TICK - 1, loc="upper left", framealpha=0.95)

    # Annotate "above y=x → over-accepts; below → over-rejects"
    ax2.text(0.95, 0.05, "above line: over-accepts\nbelow line: over-rejects",
             transform=ax2.transAxes, fontsize=TICK - 2, ha="right", va="bottom",
             bbox=dict(facecolor="white", alpha=0.85, edgecolor="gray"))

    fig.suptitle("Calibration convergence (left) + Bias direction (right)",
                 fontsize=TSIZE, y=1.02, fontweight="bold")

    out_pdf = OUT / "fig_calibration_bias.pdf"
    out_png = OUT / "fig_calibration_bias.png"
    plt.savefig(out_pdf, dpi=200, bbox_inches="tight", pad_inches=0.2)
    plt.savefig(out_png, dpi=150, bbox_inches="tight", pad_inches=0.2)
    plt.close()
    print(f"saved {out_pdf}, {out_png}")


def main():
    cells, arxiv, venues, ax_n_test = fig1_test_results()
    fig2_calibration_and_bias(cells)
    fig_validation_summary(arxiv, venues, ax_n_test)


if __name__ == "__main__":
    main()
