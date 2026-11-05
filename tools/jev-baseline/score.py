"""Score the Jev baseline the way the paper's main results table does.

Jev's raw probabilities sit mostly below 0.5, so we fit one accept cutoff per dataset on the balanced
validation set (the cutoff that maximizes balanced accuracy) and apply it to the balanced test set.

Reports, on the balanced test set: Acc (balanced accuracy), AUC, R_A/R_R (accept / reject recall),
p_A (share of papers called Accept) and, when --data_root is given, rho_R (Spearman correlation between
Jev's probability and the reviewer-rating percentile; ICLR only, where every paper has ratings).

    python score.py --jev_dir outputs [--data_root ../../paperscorer-training-and-inference/data]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from run_jev import SETS


def auc_score(golds: list[int], scores: list[float]) -> float | None:
    """Mann-Whitney AUC with tied scores sharing their average rank."""
    pairs = sorted(zip(scores, golds))
    n_pos = sum(y for _, y in pairs)
    n_neg = len(pairs) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    rank_sum_pos, rank, i = 0.0, 1, 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        rank_sum_pos += (rank + rank + (j - i) - 1) / 2 * sum(y for _, y in pairs[i:j])
        rank += j - i
        i = j
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda k: values[k])
    out, rank, i = [0.0] * len(values), 1, 0
    while i < len(order):
        j = i
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        for k in range(i, j):
            out[order[k]] = (rank + rank + (j - i) - 1) / 2
        rank += j - i
        i = j
    return out


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3:
        return None
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else None


def decision_metrics(records: list[dict], cutoff: float) -> dict:
    pos = [r for r in records if r["gold"] == 1]
    neg = [r for r in records if r["gold"] == 0]
    r_a = sum(r["score"] >= cutoff for r in pos) / len(pos)
    r_r = sum(r["score"] < cutoff for r in neg) / len(neg)
    return {"n": len(records), "bacc": (r_a + r_r) / 2, "accept_recall": r_a, "reject_recall": r_r,
            "pred_accept_rate": sum(r["score"] >= cutoff for r in records) / len(records),
            "auc": auc_score([r["gold"] for r in records], [r["score"] for r in records])}


def fit_cutoff(val: list[dict]) -> tuple[float, float]:
    """Smallest observed score whose cutoff maximizes balanced accuracy on validation."""
    best_bacc, best_cut = -1.0, 0.5
    for cutoff in sorted({r["score"] for r in val}):
        bacc = decision_metrics(val, cutoff)["bacc"]
        if bacc > best_bacc:
            best_bacc, best_cut = bacc, cutoff
    return best_cut, best_bacc


def load(jev_dir: Path, set_name: str, data_root: Path | None) -> tuple[list[dict], int]:
    data = json.loads((data_root / SETS[set_name] / "data.json").read_text()) if data_root else None
    records, failed = [], 0
    for line in (jev_dir / f"{set_name}.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row.get("noul") is None:
            failed += 1
            continue
        rating = (data[row["idx"]].get("_metadata") or {}).get("pct_rating") if data else None
        records.append({"gold": int(row["gold"]), "score": float(row["noul"]), "rating": rating})
    return records, failed


def pct(x: float) -> int:
    """Whole percent, as in the paper's R_A/R_R column."""
    return round(100 * x)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jev_dir", type=Path, default=Path(__file__).resolve().parent / "outputs")
    ap.add_argument("--data_root", type=Path, default=None, help="optional; needed only for rho_R")
    a = ap.parse_args()
    print(f"{'dataset':<8} {'cutoff':>6} {'Acc':>5} {'AUC':>5} {'R_A/R_R':>8} {'p_A':>5} {'rho_R':>6} {'n_test':>6} {'failed':>6}")
    for dom, name in (("iclr", "ICLR"), ("arxiv", "arXiv")):
        val, _ = load(a.jev_dir, f"{dom}_bal_val", a.data_root)
        test, failed = load(a.jev_dir, f"{dom}_bal_test", a.data_root)
        cutoff, _ = fit_cutoff(val)
        m = decision_metrics(test, cutoff)
        rated = [r for r in test if r["rating"] is not None]
        rho = spearman([r["score"] for r in rated], [r["rating"] for r in rated]) if dom == "iclr" else None
        rho_s = "-" if rho is None else f"{rho:+.2f}".replace("0.", ".")
        print(f"{name:<8} {cutoff:>6.2f} {100 * m['bacc']:>5.1f} {m['auc']:>5.3f} "
              f"{pct(m['accept_recall']):>4}/{pct(m['reject_recall']):<3} "
              f"{100 * m['pred_accept_rate']:>5.1f} {rho_s:>6} {m['n']:>6} {failed:>6}")


if __name__ == "__main__":
    main()
