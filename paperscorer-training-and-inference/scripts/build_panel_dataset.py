#!/usr/bin/env python3
"""Phase 2: build panel-variant data.json files from the source vision dataset.

For each split (train/val/test), iterate the vision-labelfix-filtered24480
dataset and for each entry:
  - keep _metadata + conversations as-is
  - replace `images` with the single panel PNG path (data/images_panel/<sid>.png)
  - skip entries whose panel PNG doesn't exist (logged as `missing`)

Outputs go to:
  data/iclr_..._panel_{train,validation,test}/data.json

Then registers the three new dataset names in data/dataset_info.json by
mirroring the source vision entries.

Usage:
    uv run python scripts/build_panel_dataset.py
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATASET_INFO = DATA / "dataset_info.json"

SPLITS = ["train", "validation", "test"]

SOURCES = {
    "iclr": {
        "vision_base": "iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_labelfix_v7_filtered_filtered24480",
        "sid_key": "submission_id",
        "panel_dir_name": "images_panel",
    },
    "arxiv": {
        "vision_base": "arxiv_50_50_21k_vision_wmetadata_filtered24480",
        "sid_key": "arxiv_id",
        "panel_dir_name": "images_panel_arxiv",
    },
}


_IMAGE_RUN = re.compile(r"(?:<image>\s*)+")


def collapse_image_tokens(text: str) -> str:
    """Collapse any consecutive run of <image> tokens (with whitespace) to one <image>."""
    return _IMAGE_RUN.sub("<image>", text)


def transform_entry(entry: dict, sid_key: str, panel_dir: Path, panel_dir_name: str) -> dict | None:
    sid = entry["_metadata"][sid_key]
    panel_path = panel_dir / f"{sid}.png"
    if not panel_path.exists():
        return None
    out = dict(entry)
    out["conversations"] = [
        {**c, "value": collapse_image_tokens(c["value"])} if c["from"] == "human" else c
        for c in entry["conversations"]
    ]
    out["images"] = [f"data/{panel_dir_name}/{sid}.png"]
    return out


def build_split(source: str, split: str) -> tuple[int, int]:
    src_cfg = SOURCES[source]
    vision_base = src_cfg["vision_base"]
    panel_base = f"{vision_base}_panel"
    panel_dir = DATA / src_cfg["panel_dir_name"]

    src = DATA / f"{vision_base}_{split}" / "data.json"
    dst_dir = DATA / f"{panel_base}_{split}"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / "data.json"

    src_entries = json.loads(src.read_text())
    out_entries = []
    missing = 0
    for e in src_entries:
        t = transform_entry(e, src_cfg["sid_key"], panel_dir, src_cfg["panel_dir_name"])
        if t is None:
            missing += 1
        else:
            out_entries.append(t)

    dst.write_text(json.dumps(out_entries))
    print(f"  {split}: wrote {len(out_entries)} entries to {dst.relative_to(ROOT)} "
          f"(missing panel PNG: {missing})")
    return len(out_entries), missing


def register_in_dataset_info(source: str) -> None:
    src_cfg = SOURCES[source]
    vision_base = src_cfg["vision_base"]
    panel_base = f"{vision_base}_panel"
    info = json.loads(DATASET_INFO.read_text())
    template = info[f"{vision_base}_train"]
    for split in SPLITS:
        name = f"{panel_base}_{split}"
        entry = dict(template)
        entry["file_name"] = f"{panel_base}_{split}/data.json"
        info[name] = entry
        print(f"  registered: {name}")
    DATASET_INFO.write_text(json.dumps(info, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="iclr", choices=list(SOURCES))
    ap.add_argument("--splits", nargs="+", default=SPLITS, choices=SPLITS)
    ap.add_argument("--skip_register", action="store_true",
                    help="Don't write dataset_info.json (data.json files only)")
    args = ap.parse_args()

    print(f"Building [{args.source}] panel data.json files:")
    totals = {}
    for split in args.splits:
        n, miss = build_split(args.source, split)
        totals[split] = (n, miss)

    if not args.skip_register:
        print("\nRegistering in dataset_info.json:")
        register_in_dataset_info(args.source)

    print("\nSummary:")
    for split, (n, miss) in totals.items():
        print(f"  {split}: {n} entries (missing: {miss})")


if __name__ == "__main__":
    main()
