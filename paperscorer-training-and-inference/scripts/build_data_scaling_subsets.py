"""Build per-year stratified, 50/50-balanced training subsets at 50% and 75% scale.

For each modality (text, vision):
  - Group examples by (year, label)
  - Per year, sample floor(min(accept_y, reject_y) * fraction) from each class
  - Use deterministic per-bucket seeds so 50% subset is strictly contained in 75% subset
  - Final whole-subset shuffle so the file isn't grouped by year/label

Writes new directories under data/ that mirror the existing _train layout, ready
to be referenced from data/dataset_info.json.
"""

import pathlib
import json
import os
import random
from collections import defaultdict

DATA_ROOT = str(pathlib.Path(__file__).resolve().parents[1] / "data")

MODALITIES = {
    "text": {
        "src_dir": "iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_train",
    },
    "vision": {
        "src_dir": "iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_labelfix_v7_filtered_filtered24480_train",
    },
}

FRACTIONS = [("50pct", 0.5), ("75pct", 0.75)]


def get_label(ex):
    for c in ex["conversations"]:
        if c.get("from") == "gpt":
            v = c["value"].lower()[:200]
            if "accept" in v:
                return "accept"
            if "reject" in v:
                return "reject"
    raise ValueError(f"No label found in example: {ex.get('_metadata', {}).get('submission_id')}")


def get_year(ex):
    return ex["_metadata"]["year"]


def get_sid(ex):
    return ex["_metadata"]["submission_id"]


def build_subsets(modality, src_dir):
    src_path = os.path.join(DATA_ROOT, src_dir, "data.json")
    print(f"\n=== {modality} ===")
    print(f"Loading {src_path}")
    with open(src_path) as f:
        data = json.load(f)
    print(f"Total examples: {len(data)}")

    # Bucket by (year, label)
    buckets = defaultdict(list)
    for ex in data:
        buckets[(get_year(ex), get_label(ex))].append(ex)

    years = sorted({y for (y, _) in buckets.keys()})
    print(f"Years: {years}")
    print("Per-year (accept, reject) counts:")
    for y in years:
        a = len(buckets[(y, "accept")])
        r = len(buckets[(y, "reject")])
        print(f"  {y}: accept={a}  reject={r}")

    # Independently shuffle each bucket with a deterministic seed
    for (y, lbl), bucket in buckets.items():
        seed = abs(hash((modality, y, lbl, 42))) & 0xFFFFFFFF
        random.Random(seed).shuffle(bucket)

    # For each fraction, slice each bucket and concatenate
    subsets = {}  # frac_name -> list of examples
    per_year_targets = {}  # frac_name -> {year: per_class_count}
    for frac_name, frac in FRACTIONS:
        chosen = []
        targets = {}
        for y in years:
            cap = min(len(buckets[(y, "accept")]), len(buckets[(y, "reject")]))
            per_class = int(cap * frac)  # floor
            targets[y] = per_class
            chosen.extend(buckets[(y, "accept")][:per_class])
            chosen.extend(buckets[(y, "reject")][:per_class])
        # Final whole-subset shuffle so file isn't grouped by (year, label)
        final_seed = abs(hash((modality, frac_name, "final", 7))) & 0xFFFFFFFF
        random.Random(final_seed).shuffle(chosen)
        subsets[frac_name] = chosen
        per_year_targets[frac_name] = targets

    # Print per-year targets
    print("\nPer-year per-class targets:")
    print(f"  {'year':<6} {'50% pc':<10} {'50% total':<12} {'75% pc':<10} {'75% total':<12}")
    for y in years:
        t50 = per_year_targets["50pct"][y]
        t75 = per_year_targets["75pct"][y]
        print(f"  {y:<6} {t50:<10} {2 * t50:<12} {t75:<10} {2 * t75:<12}")
    print(f"  {'TOTAL':<6} {'':<10} {sum(2 * v for v in per_year_targets['50pct'].values()):<12} "
          f"{'':<10} {sum(2 * v for v in per_year_targets['75pct'].values()):<12}")

    # Sanity-check counts and balance
    for frac_name, examples in subsets.items():
        n = len(examples)
        n_acc = sum(1 for ex in examples if get_label(ex) == "accept")
        n_rej = n - n_acc
        print(f"\n[{modality} {frac_name}] total={n}  accept={n_acc}  reject={n_rej}")
        assert n_acc == n_rej, f"{modality} {frac_name}: not 50/50 ({n_acc} vs {n_rej})"
        # Per-year balance
        per_year = defaultdict(lambda: [0, 0])
        for ex in examples:
            idx = 0 if get_label(ex) == "accept" else 1
            per_year[get_year(ex)][idx] += 1
        for y in sorted(per_year):
            a, r = per_year[y]
            assert a == r, f"{modality} {frac_name} year {y}: {a} accept vs {r} reject"

    # Nesting check: 50% submission ids ⊂ 75% submission ids
    sids_50 = {get_sid(ex) for ex in subsets["50pct"]}
    sids_75 = {get_sid(ex) for ex in subsets["75pct"]}
    assert sids_50.issubset(sids_75), f"{modality}: 50% is NOT a strict subset of 75%"
    print(f"[{modality}] nesting check passed: 50% ({len(sids_50)}) ⊂ 75% ({len(sids_75)})")

    # Write to disk
    for frac_name, examples in subsets.items():
        out_dir = os.path.join(DATA_ROOT, f"{src_dir}_{frac_name}")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "data.json")
        with open(out_path, "w") as f:
            json.dump(examples, f)
        print(f"Wrote {out_path} ({len(examples)} examples)")


def main():
    for modality, info in MODALITIES.items():
        build_subsets(modality, info["src_dir"])
    print("\nAll subsets built successfully.")


if __name__ == "__main__":
    main()
