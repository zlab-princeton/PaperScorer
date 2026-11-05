"""Build merged-on-disk training datasets: balanced + residual/max-reject pools.

Produces 4 merged datasets registered in data/dataset_info.json:
  text_balanced_sampler_iclr_merged_train
  vision_balanced_sampler_iclr_merged_train
  text_balanced_sampler_arxiv_merged_train
  vision_balanced_sampler_arxiv_merged_train

Each is the concat of (balanced source) + (residual/max-reject source). The
accept_reject_label field is preserved if already injected by
scripts/add_accept_reject_label.py; otherwise this script will inject it on the fly.

Usage:
  python scripts/build_balanced_sampler_merged.py [--dry_run]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_INFO = REPO_ROOT / "data" / "dataset_info.json"
BOXED_LABEL = re.compile(r"\\boxed\{(Accept|Reject)\}")


def derive_label(rec):
    if rec.get("_metadata") and rec["_metadata"].get("answer") in ("Accept", "Reject"):
        return 1 if rec["_metadata"]["answer"] == "Accept" else 0
    convo = rec.get("conversations") or []
    if convo:
        m = BOXED_LABEL.search(convo[-1].get("value", ""))
        if m:
            return 1 if m.group(1) == "Accept" else 0
    return None


def shard_paths(file_name):
    path = REPO_ROOT / "data" / file_name
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(path.glob("data*.json"))
    parent = path.parent
    return sorted(parent.glob(path.name + "*.json"))


def load_records(info, key):
    if key not in info:
        raise SystemExit(f"ERROR: dataset '{key}' not in dataset_info.json")
    entry = info[key]
    files = shard_paths(entry["file_name"])
    if not files:
        raise SystemExit(f"ERROR: no shards found for {key}")
    records = []
    for f in files:
        with open(f) as fp:
            records.extend(json.load(fp))
    return records, entry


# (merged_key, source_a, source_b, has_images)
RECIPES = [
    (
        "text_balanced_sampler_iclr_merged_train",
        "iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_train",
        "iclr_2020_2023_2025_2026_max_rejects_original_text_v7_filtered_filtered24480_train",
        False,
    ),
    (
        "vision_balanced_sampler_iclr_merged_train",
        "iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_labelfix_v7_filtered_filtered24480_train",
        "iclr_2020_2023_2025_2026_max_rejects_original_vision_v7_filtered_filtered24480_train",
        True,
    ),
    (
        "text_balanced_sampler_arxiv_merged_train",
        "arxiv_50_50_balanced_per_venue_text_wmetadata_filtered24480_train",
        "arxiv_residual_text_wmetadata_filtered24480_train",
        False,
    ),
    (
        "vision_balanced_sampler_arxiv_merged_train",
        "arxiv_50_50_balanced_per_venue_vision_wmetadata_filtered24480_train",
        "arxiv_residual_vision_wmetadata_filtered24480_train",
        True,
    ),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    info = json.load(open(DATASET_INFO))

    for merged_key, src_a_key, src_b_key, has_images in RECIPES:
        print(f"\n=== {merged_key} ===")
        print(f"  source A: {src_a_key}")
        print(f"  source B: {src_b_key}")

        # Resolve src_a fallbacks for iclr vision (multiple variants)
        if src_a_key not in info and "vision_v7_filtered" in src_a_key:
            for cand in [
                "iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_v7_filtered_train",
                "iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_labelfix_v7_filtered_filtered24480_cleaned_textconf_train",
            ]:
                if cand in info:
                    print(f"  (resolved src A -> {cand})")
                    src_a_key = cand
                    break
        if src_a_key not in info or src_b_key not in info:
            print(f"  SKIP: one of the source datasets is not yet registered (sync/filter pending)")
            continue

        records_a, entry_a = load_records(info, src_a_key)
        records_b, entry_b = load_records(info, src_b_key)
        print(f"  loaded {len(records_a)} + {len(records_b)} = {len(records_a)+len(records_b)} records")

        merged = []
        missing = 0
        a_counts = [0, 0]
        b_counts = [0, 0]
        for r in records_a:
            lbl = r.get("accept_reject_label")
            if lbl is None:
                lbl = derive_label(r)
            if lbl is None:
                missing += 1
                continue
            r2 = dict(r)
            r2["accept_reject_label"] = lbl
            r2["_source"] = "balanced"
            merged.append(r2)
            a_counts[lbl] += 1
        for r in records_b:
            lbl = r.get("accept_reject_label")
            if lbl is None:
                lbl = derive_label(r)
            if lbl is None:
                missing += 1
                continue
            r2 = dict(r)
            r2["accept_reject_label"] = lbl
            r2["_source"] = "residual"
            merged.append(r2)
            b_counts[lbl] += 1

        n = len(merged)
        n_accept = a_counts[1] + b_counts[1]
        n_reject = a_counts[0] + b_counts[0]
        print(f"  merged rows: {n} (accept={n_accept} reject={n_reject})  missing_label_skipped={missing}")
        print(f"    source A: accept={a_counts[1]} reject={a_counts[0]}")
        print(f"    source B: accept={b_counts[1]} reject={b_counts[0]}")

        if args.dry_run:
            print("  --dry_run: not writing.")
            continue

        # Symlink-based merge to avoid duplicating source files on disk.
        # The merged "dataset" is a directory containing symlinks to source-A and
        # source-B shards, named data_part1..N.json in concat-order so HF datasets'
        # glob(data*.json) loads them in source-A-then-source-B order — which
        # matches the order our sampler reads labels.
        out_dir = REPO_ROOT / "data" / merged_key
        out_dir.mkdir(parents=True, exist_ok=True)
        for old in out_dir.iterdir():
            old.unlink()

        a_shards = shard_paths(info[src_a_key]["file_name"])
        b_shards = shard_paths(info[src_b_key]["file_name"])
        ordered = a_shards + b_shards
        for idx, src in enumerate(ordered, start=1):
            link_path = out_dir / f"data_part{idx:02d}.json"
            os.symlink(src.resolve(), link_path)
            print(f"  link {link_path.name} -> {src.relative_to(REPO_ROOT)}")

        # Register in dataset_info.json — file_name is the dir (HF globs data*.json)
        new_entry = {
            "file_name": merged_key,
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "accept_reject_label": "accept_reject_label",
            },
            "tags": {
                "role_tag": "from",
                "content_tag": "value",
                "user_tag": "human",
                "assistant_tag": "gpt",
                "system_tag": "system",
            },
        }
        if has_images:
            new_entry["columns"]["images"] = "images"
        info[merged_key] = new_entry

    if not args.dry_run:
        bak = DATASET_INFO.with_suffix(".json.bak.build_balanced_sampler_merged")
        if not bak.exists():
            shutil.copy(DATASET_INFO, bak)
            print(f"\n  backed up dataset_info.json -> {bak.name}")
        with open(DATASET_INFO.with_suffix(".json.tmp"), "w") as fp:
            json.dump(info, fp, indent=2)
        os.replace(DATASET_INFO.with_suffix(".json.tmp"), DATASET_INFO)
        print(f"  wrote {DATASET_INFO}")

    print("\nDone.")


if __name__ == "__main__":
    main()
