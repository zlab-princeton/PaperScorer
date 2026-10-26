#!/usr/bin/env python3
"""Build the text+panel multimodal dataset.

For each split, join the text-labelfix-v7-filtered dataset with the panel
PNGs by submission_id. For each match, append ' <image>' to the last user
turn and set images=[data/images_panel/<sid>.png]. Skips entries whose
panel PNG isn't present.

Outputs:
    data/iclr_..._text_panel_{train,validation,test}/data.json

Then registers the three new dataset names in data/dataset_info.json,
copying the columns/tags from the source vision (panel) entry so the
sharegpt loader knows how to map the `images` field.
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
        "text_base": "iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered",
        "panel_base": "iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_labelfix_v7_filtered_filtered24480_panel",
        "out_base":   "iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_panel_v7_filtered",
        "sid_key": "submission_id",
        "panel_dir_name": "images_panel",
    },
    "arxiv": {
        "text_base":  "arxiv_50_50_21k_text_wmetadata_filtered24480",
        "panel_base": "arxiv_50_50_21k_vision_wmetadata_filtered24480_panel",
        "out_base":   "arxiv_50_50_21k_text_panel_wmetadata_filtered24480",
        "sid_key": "arxiv_id",
        "panel_dir_name": "images_panel_arxiv",
    },
}


# qwen2_vl's MM plugin scans the user message for <image>/<audio>/<video> and tries
# to bind each tag to a media file. The paper markdown body occasionally contains
# bare instances of these tags (HTML-derived papers, code blocks). Scrub them to a
# bracketed form so the loader doesn't trip on stray tags before our terminal <image>.
_STRAY_MM_TAG = re.compile(r"<(image|audio|video)>", re.IGNORECASE)


def scrub_stray_mm_tags(text: str) -> str:
    return _STRAY_MM_TAG.sub(lambda m: f"[{m.group(1).lower()}]", text)


def transform_entry(entry: dict, sid_key: str, panel_dir: Path, panel_dir_name: str) -> dict | None:
    sid = entry["_metadata"][sid_key]
    panel_path = panel_dir / f"{sid}.png"
    if not panel_path.exists():
        return None
    out = dict(entry)
    new_convs = []
    last_human_idx = max(
        (i for i, c in enumerate(entry["conversations"]) if c["from"] == "human"),
        default=-1,
    )
    if last_human_idx == -1:
        return None
    for i, c in enumerate(entry["conversations"]):
        if i == last_human_idx:
            cleaned = scrub_stray_mm_tags(c["value"]).rstrip()
            new_convs.append({**c, "value": cleaned + "\n\n<image>"})
        else:
            new_convs.append(c)
    out["conversations"] = new_convs
    out["images"] = [f"data/{panel_dir_name}/{sid}.png"]
    return out


def build_split(source: str, split: str) -> tuple[int, int]:
    cfg = SOURCES[source]
    panel_dir = DATA / cfg["panel_dir_name"]

    src = DATA / f"{cfg['text_base']}_{split}" / "data.json"
    dst_dir = DATA / f"{cfg['out_base']}_{split}"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / "data.json"

    out_entries = []
    skipped = 0
    for e in json.loads(src.read_text()):
        t = transform_entry(e, cfg["sid_key"], panel_dir, cfg["panel_dir_name"])
        if t is None:
            skipped += 1
        else:
            out_entries.append(t)

    dst.write_text(json.dumps(out_entries))
    print(f"  {split}: wrote {len(out_entries)} entries to {dst.relative_to(ROOT)} (skipped: {skipped})")
    return len(out_entries), skipped


def register_in_dataset_info(source: str) -> None:
    cfg = SOURCES[source]
    info = json.loads(DATASET_INFO.read_text())
    template = info[f"{cfg['panel_base']}_train"]
    for split in SPLITS:
        name = f"{cfg['out_base']}_{split}"
        entry = dict(template)
        entry["file_name"] = f"{cfg['out_base']}_{split}/data.json"
        info[name] = entry
        print(f"  registered: {name}")
    DATASET_INFO.write_text(json.dumps(info, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="iclr", choices=list(SOURCES))
    ap.add_argument("--splits", nargs="+", default=SPLITS, choices=SPLITS)
    ap.add_argument("--skip_register", action="store_true")
    args = ap.parse_args()

    print(f"Building [{args.source}] text+panel data.json files:")
    totals = {}
    for split in args.splits:
        n, sk = build_split(args.source, split)
        totals[split] = (n, sk)

    if not args.skip_register:
        print("\nRegistering in dataset_info.json:")
        register_in_dataset_info(args.source)

    print("\nSummary:")
    for split, (n, sk) in totals.items():
        print(f"  {split}: {n} entries (skipped no-panel: {sk})")


if __name__ == "__main__":
    main()
