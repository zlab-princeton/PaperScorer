#!/usr/bin/env python3
"""
Calibrated cross-eval tables for both ICLR (3x3) and arxiv (per-venue), at ckpt 1322.

Calibration semantics:
  - ICLR: per-(train, val) cell threshold τ* maximizing accuracy on val filtered
    to year >= 2025; applied at test time which is also year-filtered to 25/26.
    Test results split per-cell into [25 acc] / [26 acc].
  - Arxiv: per-(train, venue) threshold τ*_v maximizing accuracy on val of that
    venue (y24up val, year >= 2024 already). Sparse venues (val n < 10) fall
    back to a model-global τ* over all val. Test results per-(train, venue),
    with acl/emnlp/naacl rolled into acl_family.

Score: lp_accept[k] - lp_reject[k] at the decision step k where the chosen
token's logprob equals max(lp_accept, lp_reject), with a fallback derived
from min(token_logprobs) and the chosen class.

Run:
    .venv/bin/python scripts/iclr_arxiv_calibrated_eval.py
"""
from __future__ import annotations
import pathlib

import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES  = ROOT / "results/final_sweep_v7_datasweepv3/optim_search_2026"
DATA = ROOT / "data"
CKPT = 1322

# ---------- ICLR ----------
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
ICLR_RATIOS = ["50_50", "40_60", "30_70"]
ICLR_LABEL  = {"50_50": "50/50", "40_60": "40/60", "30_70": "30/70"}

# ---------- ARXIV ----------
ARXIV_TEST_PATH = DATA / "arxiv_50_50_21k_text_wmetadata_filtered24480_y24up_test/data.json"
ARXIV_VAL_PATH  = DATA / "arxiv_50_50_21k_text_wmetadata_filtered24480_y24up_validation/data.json"
ARXIV_RES = ROOT / "results/cross_conference_arxiv_y24up"
ARXIV_RATIOS = ["50_50", "40_60", "30_70"]
ARXIV_SHORTS = {"50_50": "bz32_lr1e-6_text",
                "40_60": "bz32_lr1e-6_text_40_60",
                "30_70": "bz32_lr1e-6_text_30_70"}
ACL_FAMILY = {"acl", "emnlp", "naacl"}

# ---------- score / metric helpers ----------
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
    """argmax-accuracy τ over [(score, gold_int)]. Predict accept iff score > τ."""
    if not pairs: return 0.0
    pairs = sorted(pairs)
    n = len(pairs); npos = sum(g for _,g in pairs)
    correct = npos
    best_acc, best_tau = correct/n, -math.inf
    for s, g in pairs:
        correct += (1 if g == 0 else -1)
        if correct/n > best_acc:
            best_acc, best_tau = correct/n, s
    return best_tau

# ---------- generic loader ----------
def load_pairs(jsonl_path: Path, meta_path: Path, year_min: int = None,
               include_meta=False):
    """Returns list of (year, gold, score, *meta) for valid samples."""
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
            if include_meta:
                # try arxiv venue
                m = meta[i].get("_metadata") or {}
                venue = (m.get("pl_venue") or m.get("venue") or "?").lower()
                if venue in ACL_FAMILY: venue = "acl_family"
                cy = m.get("conference_year")
                try: cy = int(cy) if cy is not None else None
                except: pass
                out.append((yr, gold, s, venue, cy))
            else:
                out.append((yr, gold, s))
    return out

# ---------- ICLR analysis ----------
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
    # The val_calib dirs are tagged with "balanced" (not "50_50") for the
    # balanced val ratio; "40_60"/"30_70" tags match directly.
    val_tag = "balanced" if val == "50_50" else val
    return ROOT / "results/iclr_val_calib" / short / f"val_{val_tag}" / f"finetuned-ckpt-{CKPT}.jsonl"

def iclr_cell(train, test):
    test_pairs = load_pairs(iclr_test_path(train, test), ICLR_TEST[test], year_min=2025)
    val_pairs  = load_pairs(iclr_val_path(train, test),  ICLR_VAL[test],  year_min=2025)
    if not val_pairs:
        # fall back to no-year-filter val if year>=2025 has zero samples
        val_pairs = load_pairs(iclr_val_path(train, test), ICLR_VAL[test], year_min=None)
    tau = best_tau([(s,g) for _,g,s in val_pairs])
    out = {"tau": tau, "n_val_25up": len(val_pairs)}
    for yr in (2025, 2026):
        sub = [(g,s) for y,g,s in test_pairs if y == yr]
        n = len(sub)
        if n == 0:
            out[yr] = {"acc_raw": None, "acc_cal": None, "auc": None, "n": 0}
            continue
        acc_raw = sum(1 for g,s in sub if (s>0)==(g==1))/n*100
        acc_cal = sum(1 for g,s in sub if (s>tau)==(g==1))/n*100
        scores = [s for g,s in sub]; gold = [g for g,s in sub]
        out[yr] = {"acc_raw":acc_raw, "acc_cal":acc_cal, "auc":auc(scores, gold), "n":n}
    return out

def print_iclr_acc():
    print("=" * 100)
    print("ICLR 3×3 ACC (text, ckpt 1322), test year-filtered to 25/26, calibrated on val year>=2025")
    print(f"Format: raw_acc_25/raw_acc_26  →  calib_acc_25/calib_acc_26  (n25,n26)  τ*  n_val(>=25)")
    print("=" * 100)
    print(f"{'Train':<7} {'Test':<7} {'Raw 25/26':<14} {'Calib 25/26':<14} {'(n25,n26)':<14} {'τ*':>7} {'n_val':>7}")
    print("-" * 100)
    for train in ICLR_RATIOS:
        for test in ICLR_RATIOS:
            m = iclr_cell(train, test)
            r = m[2025]; s = m[2026]
            if r["acc_raw"] is None or s["acc_raw"] is None:
                print(f"{ICLR_LABEL[train]:<7} {ICLR_LABEL[test]:<7} -- missing --")
                continue
            raw = f"{r['acc_raw']:5.1f}/{s['acc_raw']:5.1f}"
            cal = f"{r['acc_cal']:5.1f}/{s['acc_cal']:5.1f}"
            n   = f"({r['n']:>4},{s['n']:>4})"
            tau = f"{m['tau']:+.2f}"
            print(f"{ICLR_LABEL[train]:<7} {ICLR_LABEL[test]:<7} {raw:<14} {cal:<14} {n:<14} {tau:>7} {m['n_val_25up']:>7}")

def print_iclr_auc():
    print()
    print("=" * 100)
    print("ICLR 3×3 AUC (text, ckpt 1322), test year-filtered to 25/26 (threshold-independent)")
    print("=" * 100)
    print(f"{'Train':<7} {'Test':<7} {'AUC 25/26':<14} {'(n25,n26)':<14}")
    print("-" * 100)
    for train in ICLR_RATIOS:
        for test in ICLR_RATIOS:
            m = iclr_cell(train, test)
            r = m[2025]; s = m[2026]
            au25 = f"{r['auc']:.3f}" if r["auc"] is not None else "  N/A"
            au26 = f"{s['auc']:.3f}" if s["auc"] is not None else "  N/A"
            n   = f"({r['n']:>4},{s['n']:>4})"
            print(f"{ICLR_LABEL[train]:<7} {ICLR_LABEL[test]:<7} {au25}/{au26}     {n:<14}")

# ---------- ARXIV analysis ----------
def arxiv_jsonl(short, kind):
    return ARXIV_RES / short / f"arxiv_{kind}" / f"finetuned-ckpt-{CKPT}.jsonl"

def arxiv_collect():
    """Load all (val + test) data per train ratio with venue annotations."""
    data = {}
    for ratio in ARXIV_RATIOS:
        short = ARXIV_SHORTS[ratio]
        val_pairs  = load_pairs(arxiv_jsonl(short, "val"),  ARXIV_VAL_PATH,  include_meta=True)
        test_pairs = load_pairs(arxiv_jsonl(short, "eval"), ARXIV_TEST_PATH, include_meta=True)
        # Group by venue
        val_by_v  = defaultdict(list)
        test_by_v = defaultdict(list)
        for _,g,s,v,_ in val_pairs:  val_by_v[v].append((s,g))
        for _,g,s,v,_ in test_pairs: test_by_v[v].append((s,g))
        # Compute per-venue τ (with sparse fallback)
        global_tau = best_tau([(s,g) for _,g,s,*_ in val_pairs])
        thresholds = {}
        for v, pairs in val_by_v.items():
            if len(pairs) < 10:
                thresholds[v] = (global_tau, "global")
            else:
                thresholds[v] = (best_tau(pairs), "venue")
        # Also venues only present in test (no val): use global
        for v in test_by_v:
            thresholds.setdefault(v, (global_tau, "global"))
        data[ratio] = {
            "val_by_v": val_by_v, "test_by_v": test_by_v,
            "thresholds": thresholds, "global_tau": global_tau,
            "test_total": sum(len(p) for p in test_by_v.values()),
            "val_total":  sum(len(p) for p in val_by_v.values()),
        }
    return data

def print_arxiv_acc(data):
    venues = sorted({v for ratio in ARXIV_RATIOS for v in data[ratio]["test_by_v"]})
    print()
    print("=" * 130)
    print("ARXIV ACC (text, ckpt 1322), y24up test (year≥2024), per-venue τ* from y24up val (sparse→global)")
    print("Format per cell: raw_acc / calib_acc  (n)")
    print("=" * 130)
    header = f"{'Venue':<12}" + "".join(f"  {ICLR_LABEL[r]:^22}" for r in ARXIV_RATIOS)
    print(header)
    print("-" * 130)
    for v in venues:
        row = f"{v:<12}"
        for ratio in ARXIV_RATIOS:
            tp = data[ratio]["test_by_v"].get(v, [])
            tau, src = data[ratio]["thresholds"].get(v, (data[ratio]["global_tau"], "global"))
            n = len(tp)
            if n == 0:
                row += f"  {'(no test data)':^22}"
                continue
            acc_raw = sum(1 for s,g in tp if (s>0)==(g==1))/n*100
            acc_cal = sum(1 for s,g in tp if (s>tau)==(g==1))/n*100
            cell = f"{acc_raw:4.1f}/{acc_cal:4.1f}  (n={n:>3}{'g' if src=='global' else ''})"
            row += f"  {cell:^22}"
        print(row)
    print("-" * 130)
    # Overall row
    row = f"{'OVERALL':<12}"
    for ratio in ARXIV_RATIOS:
        all_tp = [p for v_pairs in data[ratio]["test_by_v"].values() for p in v_pairs]
        # raw uses τ=0 globally; calibrated uses per-venue τ
        n = len(all_tp)
        acc_raw = sum(1 for s,g in all_tp if (s>0)==(g==1))/n*100 if n else 0
        # for calibrated, need to apply per-venue τ — recompute
        n_calib_correct = 0
        for v, tp in data[ratio]["test_by_v"].items():
            tau, _ = data[ratio]["thresholds"].get(v, (data[ratio]["global_tau"], "global"))
            n_calib_correct += sum(1 for s,g in tp if (s>tau)==(g==1))
        acc_cal = n_calib_correct/n*100 if n else 0
        cell = f"{acc_raw:4.1f}/{acc_cal:4.1f}  (n={n})"
        row += f"  {cell:^22}"
    print(row)

def print_arxiv_auc(data):
    venues = sorted({v for ratio in ARXIV_RATIOS for v in data[ratio]["test_by_v"]})
    print()
    print("=" * 130)
    print("ARXIV AUC (text, ckpt 1322), y24up test, per-venue (threshold-independent)")
    print("=" * 130)
    header = f"{'Venue':<12}" + "".join(f"  {ICLR_LABEL[r]:^14}" for r in ARXIV_RATIOS)
    print(header)
    print("-" * 130)
    for v in venues:
        row = f"{v:<12}"
        for ratio in ARXIV_RATIOS:
            tp = data[ratio]["test_by_v"].get(v, [])
            n = len(tp)
            if n == 0 or len({g for _,g in tp}) < 2:
                row += f"  {'  N/A':^14}"
                continue
            scores = [s for s,_ in tp]; gold = [g for _,g in tp]
            au = auc(scores, gold)
            cell = f"{au:.3f} (n={n:>3})"
            row += f"  {cell:^14}"
        print(row)
    print("-" * 130)
    row = f"{'OVERALL':<12}"
    for ratio in ARXIV_RATIOS:
        all_tp = [p for v_pairs in data[ratio]["test_by_v"].values() for p in v_pairs]
        scores = [s for s,_ in all_tp]; gold = [g for _,g in all_tp]
        au = auc(scores, gold)
        cell = f"{au:.3f} (n={len(all_tp)})"
        row += f"  {cell:^14}"
    print(row)


def main():
    print_iclr_acc()
    print_iclr_auc()
    data = arxiv_collect()
    print_arxiv_acc(data)
    print_arxiv_auc(data)


if __name__ == "__main__":
    main()
