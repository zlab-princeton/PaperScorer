#!/usr/bin/env python3
"""
Per-venue threshold calibration on the y24up validation set, applied to the
y24up test set. Prints a results table with raw vs calibrated accuracy.

Score: lp_accept[5] - lp_reject[5] (log-odds at the decision step).
Threshold rule per (model, venue):
    pick tau in {scores ∪ {-inf, +inf}} that maximizes val accuracy on that venue
Sparse venues (val n < SPARSE_N) fall back to the model's global threshold
(computed on all val samples pooled).

Venue rollup: acl + emnlp + naacl => "acl_family" (per user note: same
conference family treated as one for threshold purposes).
"""
from __future__ import annotations
import pathlib
import json
import math
from collections import defaultdict
from pathlib import Path

# ---------- config ----------
ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/cross_conference_arxiv_y24up"
DATA = ROOT / "data"

VAL_DS  = "arxiv_50_50_21k_text_wmetadata_filtered24480_y24up_validation"
TEST_DS = "arxiv_50_50_21k_text_wmetadata_filtered24480_y24up_test"

MODELS = [
    ("50/50", "bz32_lr1e-6_text"),
    ("40/60", "bz32_lr1e-6_text_40_60"),
    ("30/70", "bz32_lr1e-6_text_30_70"),
]
CKPT = 1322

SPARSE_N = 10  # venues with < this many val samples → use global threshold

ACL_FAMILY = {"acl", "emnlp", "naacl"}


def venue_of(meta: dict) -> str:
    v = (meta.get("pl_venue") or meta.get("venue") or "?").lower()
    return "acl_family" if v in ACL_FAMILY else v


# ---------- score extraction ----------

def score_logodds(row: dict) -> float | None:
    pred = row.get("predict", "")
    chosen = 1 if "Accept" in pred else (0 if "Reject" in pred else None)
    if chosen is None:
        return None
    la = row.get("logprob_accept") or []
    lr = row.get("logprob_reject") or []
    tl = row.get("token_logprobs") or []
    for k in range(min(len(tl), len(la), len(lr))):
        if la[k] is not None and lr[k] is not None and tl[k] is not None:
            if math.isclose(tl[k], max(la[k], lr[k]), abs_tol=1e-3):
                return la[k] - lr[k]
    return 6.0 if chosen == 1 else -6.0


def gold_of(row: dict) -> int | None:
    lab = row.get("label", "")
    if "Accept" in lab: return 1
    if "Reject" in lab: return 0
    return None


# ---------- loaders ----------

def load_pairs(jsonl_path: Path, meta_list: list[dict]) -> list[tuple[float, int, str]]:
    """Returns list of (score, gold, venue_key) for each sample."""
    out = []
    with jsonl_path.open() as fh:
        for i, line in enumerate(fh):
            if i >= len(meta_list): break
            r = json.loads(line)
            s = score_logodds(r)
            g = gold_of(r)
            if s is None or g is None: continue
            v = venue_of(meta_list[i].get("_metadata", {}))
            out.append((s, g, v))
    return out


# ---------- threshold calibration ----------

def best_threshold(pairs: list[tuple[float, int]]) -> tuple[float, float]:
    """Return (best_tau, best_acc) maximizing acc on `pairs` of (score, gold).
    Predicts Accept iff score > tau."""
    if not pairs:
        return 0.0, 0.0
    pairs = sorted(pairs, key=lambda p: p[0])
    n = len(pairs)
    n_pos = sum(g for _, g in pairs)

    # Try threshold below all (predict all Accept) and threshold above each value
    # Predict Accept iff score > tau.
    #   tau = -inf: all Accept → correct = n_pos
    #   tau just above scores[i]: items 0..i predicted Reject, i+1..n-1 Accept
    correct = n_pos  # tau = -inf
    best_acc = correct / n
    best_tau = -math.inf
    cum_pos_left = 0  # number of positives we'd reject by raising tau past idx
    cum_neg_left = 0  # number of negatives we correctly reject by raising tau
    for i, (s, g) in enumerate(pairs):
        if g == 1:
            correct -= 1  # this positive moves from Accept-correct to Reject-wrong
        else:
            correct += 1  # this negative moves from Accept-wrong to Reject-correct
        # tau = s (predicting score > s as Accept, score == s also Reject)
        acc = correct / n
        if acc > best_acc:
            best_acc = acc
            # use a tau slightly above s so equal scores fall on Reject
            next_s = pairs[i+1][0] if i+1 < n else s + 1.0
            best_tau = s if next_s > s else s
    return best_tau, best_acc


# ---------- main ----------

def main():
    val_meta  = json.loads((DATA / VAL_DS  / "data.json").read_text())
    test_meta = json.loads((DATA / TEST_DS / "data.json").read_text())

    print(f"=== Per-venue threshold calibration ===")
    print(f"  val:  {len(val_meta)} samples ({VAL_DS})")
    print(f"  test: {len(test_meta)} samples ({TEST_DS})")
    print(f"  sparse_n cutoff: < {SPARSE_N} val samples → use global threshold")
    print(f"  rollup: {ACL_FAMILY} → acl_family")
    print()

    rows_summary = []
    rows_per_venue = []

    for label, short in MODELS:
        val_jsonl  = RESULTS / short / "arxiv_val"  / f"finetuned-ckpt-{CKPT}.jsonl"
        test_jsonl = RESULTS / short / "arxiv_eval" / f"finetuned-ckpt-{CKPT}.jsonl"
        if not val_jsonl.exists():
            print(f"  [skip] {label}: no val jsonl at {val_jsonl}")
            continue
        if not test_jsonl.exists():
            print(f"  [skip] {label}: no test jsonl at {test_jsonl}")
            continue

        val_pairs  = load_pairs(val_jsonl,  val_meta)
        test_pairs = load_pairs(test_jsonl, test_meta)

        # Group by venue
        val_by_v  = defaultdict(list)
        test_by_v = defaultdict(list)
        for s, g, v in val_pairs:  val_by_v[v].append((s, g))
        for s, g, v in test_pairs: test_by_v[v].append((s, g))

        # Global threshold (fallback for sparse venues)
        tau_global, acc_global_val = best_threshold([(s, g) for s, g, _ in val_pairs])

        # Per-venue thresholds
        thresholds = {}
        for v, pairs in val_by_v.items():
            if len(pairs) < SPARSE_N:
                thresholds[v] = (tau_global, "global_fallback", len(pairs))
            else:
                tau, _ = best_threshold(pairs)
                thresholds[v] = (tau, "per_venue", len(pairs))

        # ---- compute metrics ----
        # Test accuracy with raw threshold (tau=0)
        test_total = len(test_pairs)
        raw_correct = sum(1 for s, g, v in test_pairs if (s > 0) == (g == 1))
        # Calibrated: per-venue threshold
        calib_correct = 0
        for s, g, v in test_pairs:
            tau, _, _ = thresholds.get(v, (tau_global, "global_fallback", 0))
            pred = 1 if s > tau else 0
            if pred == g: calib_correct += 1

        rows_summary.append({
            "model": label,
            "test_n": test_total,
            "raw_acc": raw_correct / test_total,
            "calib_acc": calib_correct / test_total,
            "tau_global": tau_global,
        })

        for v in sorted(set(val_by_v) | set(test_by_v)):
            tau, src, n_val = thresholds.get(v, (tau_global, "global_fallback", 0))
            tp_v = test_by_v.get(v, [])
            n_test = len(tp_v)
            raw_acc_v = (sum(1 for s, g in tp_v if (s > 0) == (g == 1)) / n_test) if n_test else None
            calib_acc_v = (sum(1 for s, g in tp_v if (s > tau) == (g == 1)) / n_test) if n_test else None
            rows_per_venue.append({
                "model": label, "venue": v, "n_val": n_val, "n_test": n_test,
                "tau": tau, "src": src,
                "raw_acc": raw_acc_v, "calib_acc": calib_acc_v,
            })

    # ---- print summary table ----
    print(f"=== SUMMARY: y24up text (1415 test / 722 val) ===")
    print(f"{'model':<8} {'test_n':<8} {'raw_acc':<9} {'calib_acc':<11} {'lift':<8} {'τ_global':<10}")
    for r in rows_summary:
        lift = (r["calib_acc"] - r["raw_acc"]) * 100
        print(f"{r['model']:<8} {r['test_n']:<8} {r['raw_acc']:.3f}     {r['calib_acc']:.3f}       {lift:+.1f}pp   {r['tau_global']:+.3f}")

    print()
    print(f"=== PER-VENUE breakdown ===")
    print(f"{'model':<8} {'venue':<14} {'n_val':<6} {'n_test':<7} {'τ':<8} {'src':<18} {'raw_acc':<9} {'calib_acc':<10}")
    for r in rows_per_venue:
        ra = f"{r['raw_acc']:.3f}" if r['raw_acc'] is not None else "    -"
        ca = f"{r['calib_acc']:.3f}" if r['calib_acc'] is not None else "    -"
        tau_s = f"{r['tau']:+.3f}" if r['tau'] != float("-inf") else "  -inf"
        print(f"{r['model']:<8} {r['venue']:<14} {r['n_val']:<6} {r['n_test']:<7} {tau_s:<8} {r['src']:<18} {ra:<9} {ca:<10}")


if __name__ == "__main__":
    main()
