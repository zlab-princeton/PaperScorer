"""Grade coding agent predictions against ground truth.

Usage: uv run scripts/grade_agent_predictions.py [path/to/PREDICTIONS.json]
Default predictions path: coding_agents/PREDICTIONS.json
"""

import json
import sys
from collections import Counter
from pathlib import Path

GROUND_TRUTH_PATH = Path("data/agent_ground_truth.json")
DEFAULT_PREDICTIONS_PATH = Path("coding_agents/PREDICTIONS.json")


def main():
    pred_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PREDICTIONS_PATH

    if not pred_path.exists():
        print(f"ERROR: {pred_path} not found. Has the agent finished?")
        sys.exit(1)

    with open(GROUND_TRUTH_PATH) as f:
        gt = json.load(f)
    with open(pred_path) as f:
        preds = json.load(f)

    # Check coverage
    gt_keys = set(gt.keys())
    pred_keys = set(preds.keys())
    missing = gt_keys - pred_keys
    extra = pred_keys - gt_keys
    common = gt_keys & pred_keys

    if missing:
        print(f"WARNING: {len(missing)} papers missing from predictions")
    if extra:
        print(f"WARNING: {len(extra)} extra keys in predictions not in ground truth")

    # Normalize predictions — handle both {"decision": ..., "why": ...} and flat string format
    no_why = 0
    for k in list(preds.keys()):
        v = preds[k]
        if isinstance(v, dict):
            if "why" not in v or not v.get("why", "").strip():
                no_why += 1
            preds[k] = v.get("decision", "").strip().lower()
        else:
            no_why += 1
            preds[k] = str(v).strip().lower()

    if no_why:
        print(f"WARNING: {no_why} predictions missing 'why' reasoning")

    invalid = {k: preds[k] for k in common if preds[k] not in ("accept", "reject")}
    if invalid:
        print(f"WARNING: {len(invalid)} invalid labels: {dict(list(invalid.items())[:5])}")

    # Compute metrics on common keys with valid labels
    valid_keys = [k for k in common if preds[k] in ("accept", "reject")]

    tp = fp = tn = fn = 0
    correct = 0
    total = len(valid_keys)

    for k in valid_keys:
        g, p = gt[k], preds[k]
        if g == p:
            correct += 1
        # Treat "accept" as positive
        if g == "accept" and p == "accept":
            tp += 1
        elif g == "reject" and p == "reject":
            tn += 1
        elif g == "reject" and p == "accept":
            fp += 1
        elif g == "accept" and p == "reject":
            fn += 1

    accuracy = correct / total if total else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    pred_dist = Counter(preds[k] for k in valid_keys)
    gt_dist = Counter(gt[k] for k in valid_keys)

    print(f"\n{'='*50}")
    print(f"  Agent Prediction Grading Report")
    print(f"{'='*50}")
    print(f"  Predictions file : {pred_path}")
    print(f"  Ground truth     : {GROUND_TRUTH_PATH}")
    print(f"  Total evaluated  : {total} / {len(gt_keys)}")
    if missing:
        print(f"  Missing          : {len(missing)}")
    if invalid:
        print(f"  Invalid labels   : {len(invalid)}")
    print()
    print(f"  Ground truth dist: {dict(sorted(gt_dist.items()))}")
    print(f"  Prediction dist  : {dict(sorted(pred_dist.items()))}")
    print()
    print(f"  Accuracy  : {accuracy:.1%}  ({correct}/{total})")
    print(f"  Precision : {precision:.1%}  (accept)")
    print(f"  Recall    : {recall:.1%}  (accept)")
    print(f"  F1        : {f1:.1%}  (accept)")
    print()
    print(f"  Confusion Matrix (rows=true, cols=pred):")
    print(f"                 pred_accept  pred_reject")
    print(f"  true_accept      {tp:>5}        {fn:>5}")
    print(f"  true_reject      {fp:>5}        {tn:>5}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
