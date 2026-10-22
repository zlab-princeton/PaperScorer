#!/usr/bin/env python3
"""
ICLR 3x3 (train_ratio × eval_ratio) per-cell threshold calibration on
validation 25/26, applied to test 25/26. Prints raw vs calibrated accuracy
plus the optimal τ\* for each cell.

Score: logprob_accept[5] - logprob_reject[5] (DECISION_TOKEN_IDX=5; same
convention as scripts/tmp_latex_dir/generate_calibration*.py — directly
reads the per-step accept/reject logprobs that --save_logprobs records,
no max-search).

Threshold: best_threshold from scripts/arxiv_calibrated_acc.py (sweep on
score, pick tau maximizing val accuracy).

50/50-on-balanced-val cell reuses the pre-existing
results/.../bz32_lr1e-6_text/validation-ckpt-1322.jsonl. The other 8 cells
read from results/iclr_val_calib/<short>/val_<ratio>/finetuned-ckpt-1322.jsonl.

Test cells reuse the existing cross-eval matrix from
scripts/tmp_latex_dir/generate_ratio_train_curves.py (jsonl_for resolver).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Optional

# Reuse the existing test-jsonl resolver from the figure script
sys.path.insert(0, str(Path(__file__).resolve().parent / "tmp_latex_dir"))
from generate_ratio_train_curves import jsonl_for, TEXT_TEST_DATA  # noqa: E402

# Reuse the threshold sweep from the arxiv calibrator
sys.path.insert(0, str(Path(__file__).resolve().parent))
from arxiv_calibrated_acc import best_threshold, gold_of  # noqa: E402


# --- config ----------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RESULTS_VAL_BASE = ROOT / "results/iclr_val_calib"
RESULTS_50_50_VAL_REUSED = (
    ROOT / "results/final_sweep_v7_datasweepv3/optim_search_2026"
    / "bz32_lr1e-6_text/validation-ckpt-1322.jsonl"
)

DATA = ROOT / "data"
VAL_DATA = {
    "50_50":    DATA / "iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_validation/data.json",
    "40_60":    DATA / "iclr_2020_2023_2025_2026_40_60_original_text_v7_filtered_validation_DERIVED/data.json",
    "30_70":    DATA / "iclr_2020_2023_2025_2026_30_70_original_text_v7_filtered_validation_DERIVED/data.json",
}

MODELS = [
    ("50_50", "bz32_lr1e-6_text"),
    ("40_60", "bz32_lr1e-6_text_40_60"),
    ("30_70", "bz32_lr1e-6_text_30_70"),
]
EVAL_RATIOS = ["50_50", "40_60", "30_70"]
CKPT = 1322
DECISION_TOKEN_IDX = 5
EVAL_YEARS = {2025, 2026}


# --- helpers ---------------------------------------------------------------

def val_jsonl_for(short: str, eval_ratio: str) -> Path:
    """50/50-on-balanced prefers the freshly-rerun version (with logprobs);
    falls back to the pre-existing March-24 jsonl if not yet present."""
    if short == "bz32_lr1e-6_text" and eval_ratio == "50_50":
        fresh = RESULTS_VAL_BASE / short / "val_balanced" / f"finetuned-ckpt-{CKPT}.jsonl"
        if fresh.exists():
            return fresh
        return RESULTS_50_50_VAL_REUSED
    tag = "balanced" if eval_ratio == "50_50" else eval_ratio
    return RESULTS_VAL_BASE / short / f"val_{tag}" / f"finetuned-ckpt-{CKPT}.jsonl"


def score_at_decision(rec: dict) -> Optional[float]:
    """Primary path: log-odds at index 5 from logprob_accept[5] - logprob_reject[5].
    Fallback (for old pre-`--save_logprobs` jsonls): derive log-odds from
    token_logprobs[5] (logprob of chosen token) + the predicted class."""
    la = rec.get("logprob_accept") or []
    lr = rec.get("logprob_reject") or []
    if (len(la) > DECISION_TOKEN_IDX and len(lr) > DECISION_TOKEN_IDX
            and la[DECISION_TOKEN_IDX] is not None
            and lr[DECISION_TOKEN_IDX] is not None):
        return la[DECISION_TOKEN_IDX] - lr[DECISION_TOKEN_IDX]

    # Fallback: token_logprobs[5] = logprob(chosen)
    tl = rec.get("token_logprobs") or []
    if len(tl) <= DECISION_TOKEN_IDX or tl[DECISION_TOKEN_IDX] is None:
        return None
    pred = (rec.get("predict") or "").lower()
    chose_accept = "boxed{accept}" in pred
    chose_reject = "boxed{reject}" in pred
    if not (chose_accept or chose_reject):
        return None
    p_chosen = math.exp(tl[DECISION_TOKEN_IDX])
    eps = 1e-6
    p_chosen = max(min(p_chosen, 1 - eps), eps)
    p_accept = p_chosen if chose_accept else 1.0 - p_chosen
    p_accept = max(min(p_accept, 1 - eps), eps)
    return math.log(p_accept / (1.0 - p_accept))


def load_pairs_2526(jsonl_path: Path, meta_path: Path) -> list[tuple[float, int]]:
    """Return [(score, gold)] filtered to _metadata.year in {2025, 2026}."""
    if not jsonl_path.exists():
        return []
    if not meta_path.exists():
        return []
    meta = json.loads(meta_path.read_text())
    out = []
    with jsonl_path.open() as fh:
        for i, line in enumerate(fh):
            if i >= len(meta):
                break
            yr = (meta[i].get("_metadata") or {}).get("year")
            if yr not in EVAL_YEARS:
                continue
            r = json.loads(line)
            s = score_at_decision(r)
            g = gold_of(r)
            if s is None or g is None:
                continue
            out.append((s, g))
    return out


# --- main ------------------------------------------------------------------

def main():
    print("=" * 72)
    print("ICLR text 3x3 — per-cell threshold calibration on val 25/26 → test 25/26")
    print("Score: logprob_accept[5] - logprob_reject[5] (DECISION_TOKEN_IDX=5)")
    print("=" * 72)

    cells = {}  # (train_ratio, eval_ratio) -> dict
    for train_ratio, short in MODELS:
        for eval_ratio in EVAL_RATIOS:
            val_jp = val_jsonl_for(short, eval_ratio)
            test_jp = jsonl_for("text", train_ratio, short, eval_ratio, CKPT)
            test_data = TEXT_TEST_DATA[eval_ratio]
            val_data = VAL_DATA[eval_ratio]

            val_pairs = load_pairs_2526(val_jp, val_data)
            test_pairs = load_pairs_2526(test_jp, test_data)

            cell = {
                "val_jsonl": val_jp,
                "test_jsonl": test_jp,
                "n_val": len(val_pairs),
                "n_test": len(test_pairs),
                "tau": None,
                "raw_acc": None,
                "calib_acc": None,
            }

            if val_pairs:
                tau, _ = best_threshold(val_pairs)
                cell["tau"] = tau
            if test_pairs:
                # raw: predict Accept iff score > 0 (greedy/natural threshold)
                cell["raw_acc"] = sum(1 for s, g in test_pairs if (s > 0) == (g == 1)) / len(test_pairs)
                if cell["tau"] is not None:
                    cell["calib_acc"] = sum(1 for s, g in test_pairs if (s > cell["tau"]) == (g == 1)) / len(test_pairs)

            cells[(train_ratio, eval_ratio)] = cell

    # ------------- Detailed per-cell ----------------
    print()
    print(f"{'train':<8} {'test':<8} {'n_val':>6} {'n_test':>7} {'tau*':>9} {'raw':>7} {'calib':>7} {'lift':>7}")
    print("-" * 72)
    for tr, _ in MODELS:
        for er in EVAL_RATIOS:
            c = cells[(tr, er)]
            tau_s = f"{c['tau']:+.3f}" if c["tau"] is not None else "    -"
            raw_s = f"{c['raw_acc']:.3f}" if c["raw_acc"] is not None else "  -"
            cal_s = f"{c['calib_acc']:.3f}" if c["calib_acc"] is not None else "  -"
            lift_s = (
                f"{(c['calib_acc'] - c['raw_acc']) * 100:+.2f}pp"
                if (c["raw_acc"] is not None and c["calib_acc"] is not None)
                else "  -"
            )
            print(f"{tr:<8} {er:<8} {c['n_val']:>6} {c['n_test']:>7} {tau_s:>9} {raw_s:>7} {cal_s:>7} {lift_s:>7}")

    # ------------- 3x3 SUMMARY ----------------
    print()
    print("=" * 72)
    print("SUMMARY — ICLR text 3x3 (val/test filtered to 25/26)")
    print("=" * 72)
    header = f"{'':<10}" + "  ".join(f"{'test_'+er:^21}" for er in EVAL_RATIOS)
    print(header)
    sub = f"{'':<10}" + "  ".join(f"{'raw':>5} {'calib':>5} {'τ*':>7}" for _ in EVAL_RATIOS)
    print(sub)
    for tr, _ in MODELS:
        cells_row = []
        for er in EVAL_RATIOS:
            c = cells[(tr, er)]
            raw = f"{c['raw_acc']:.3f}" if c["raw_acc"] is not None else "  -  "
            cal = f"{c['calib_acc']:.3f}" if c["calib_acc"] is not None else "  -  "
            tau = f"{c['tau']:+.3f}" if c["tau"] is not None else "   -  "
            cells_row.append(f"{raw:>5} {cal:>5} {tau:>7}")
        print(f"train_{tr:<5}" + "  ".join(cells_row))


if __name__ == "__main__":
    main()
