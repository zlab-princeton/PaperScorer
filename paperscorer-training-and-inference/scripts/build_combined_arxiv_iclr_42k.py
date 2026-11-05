#!/usr/bin/env python3
"""
Build combined arxiv-21k + iclr-21k train datasets (text and vision modalities).

Concatenates the two existing per-domain train sets verbatim (no schema
rewrites — both already use the same sharegpt SYSTEM/USER/ASSISTANT format
with identical 'You are an expert academic reviewer...' system prompt),
shuffles once with a fixed seed so training sees an interleaved mix from step
1, writes to `data/combined_arxiv_iclr_42k_{text,vision}_filtered24480_train/data.json`,
and registers both new datasets in `data/dataset_info.json` (with backup).

Run:
  python scripts/build_combined_arxiv_iclr_42k.py
"""

from __future__ import annotations

import json
import random
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
INFO_JSON = DATA / "dataset_info.json"

SEED = 42

# (modality, arxiv_source_name, iclr_source_name, new_combined_name, columns)
COMBOS = [
    (
        "text",
        "arxiv_50_50_21k_text_wmetadata_filtered24480_train",
        "iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_train",
        "combined_arxiv_iclr_42k_text_filtered24480_train",
        {"messages": "conversations"},  # no images column for text
    ),
    (
        "vision",
        "arxiv_50_50_21k_vision_wmetadata_filtered24480_train",
        "iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_labelfix_v7_filtered_filtered24480_train",
        "combined_arxiv_iclr_42k_vision_filtered24480_train",
        {"messages": "conversations", "images": "images"},
    ),
]

TAGS = {
    "role_tag": "from",
    "content_tag": "value",
    "user_tag": "human",
    "assistant_tag": "gpt",
    "system_tag": "system",
}


def resolve_source_path(info: dict, name: str) -> Path:
    """dataset_info.json's `file_name` can be either a path-to-data.json or a
    dir containing data.json. Return the actual data.json path."""
    fn = info[name]["file_name"]
    p = DATA / fn
    if p.is_dir():
        p = p / "data.json"
    return p


def load_records(path: Path) -> list:
    return json.loads(path.read_text())


def main():
    info = json.loads(INFO_JSON.read_text())

    # Backup dataset_info.json
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = INFO_JSON.with_suffix(f".json.bak.{ts}")
    shutil.copy2(INFO_JSON, bak)
    print(f"backed up dataset_info.json -> {bak.name}")

    rng = random.Random(SEED)

    new_entries = []
    for modality, arxiv_name, iclr_name, combined_name, columns in COMBOS:
        print(f"\n=== {modality} ===")
        arxiv_path = resolve_source_path(info, arxiv_name)
        iclr_path = resolve_source_path(info, iclr_name)
        print(f"  arxiv source: {arxiv_path}")
        print(f"  iclr  source: {iclr_path}")
        if not arxiv_path.exists():
            raise FileNotFoundError(arxiv_path)
        if not iclr_path.exists():
            raise FileNotFoundError(iclr_path)

        arxiv = load_records(arxiv_path)
        iclr = load_records(iclr_path)
        print(f"  arxiv records: {len(arxiv)}")
        print(f"  iclr  records: {len(iclr)}")

        combined = arxiv + iclr
        rng.shuffle(combined)
        print(f"  combined (shuffled, seed={SEED}): {len(combined)} records")

        out_dir = DATA / combined_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "data.json"
        # Atomic write: tmp + rename
        tmp_path = out_dir / "data.json.tmp"
        tmp_path.write_text(json.dumps(combined, ensure_ascii=False))
        tmp_path.rename(out_path)
        print(f"  wrote: {out_path} ({out_path.stat().st_size // (1024*1024)} MiB)")

        info[combined_name] = {
            "file_name": f"{combined_name}/data.json",
            "formatting": "sharegpt",
            "columns": columns,
            "tags": TAGS,
        }
        new_entries.append(combined_name)

    # Atomic write of dataset_info.json (tmp + rename)
    tmp = INFO_JSON.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(info, indent=2))
    tmp.rename(INFO_JSON)
    print(f"\nUPDATED {INFO_JSON.name} with {len(new_entries)} new entries:")
    for n in new_entries:
        print(f"  + {n}")


if __name__ == "__main__":
    main()
