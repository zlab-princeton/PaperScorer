#!/usr/bin/env python3
"""Build reversed-order ablation datasets for sequential-cue analysis.

For each source dataset:
  - text: reverse `# HEADER` blocks in the human turn (preamble preserved).
  - vision: reverse the `images` list (prompt's `<image>` tokens are positional).

Source data lives at  data/<NAME>/data.json
Output                data/<NAME>_reversed/data.json
GPT (assistant) turn is never modified.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SOURCES: list[tuple[str, str]] = [
    ("iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_train",                       "text"),
    ("iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_validation",                  "text"),
    ("iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_test",                        "text"),
    ("iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_labelfix_v7_filtered_filtered24480_train",       "vision"),
    ("iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_labelfix_v7_filtered_filtered24480_validation",  "vision"),
    ("iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_labelfix_v7_filtered_filtered24480_test",        "vision"),
]

_HEADER_LINE = re.compile(r'^# ', flags=re.MULTILINE)
_HEADER_SPLIT = re.compile(r'\n(?=# )')


def reverse_text_sections(human_value: str) -> str:
    m = _HEADER_LINE.search(human_value)
    if not m:
        return human_value
    preamble = human_value[: m.start()]
    body = human_value[m.start():]
    sections = _HEADER_SPLIT.split(body)
    sections.reverse()
    return preamble + '\n'.join(sections)


def reverse_text_entry(entry: dict) -> dict:
    new = dict(entry)
    new["conversations"] = [
        {**msg, "value": reverse_text_sections(msg["value"])} if msg.get("from") == "human" else msg
        for msg in entry["conversations"]
    ]
    return new


def reverse_vision_entry(entry: dict) -> dict:
    new = dict(entry)
    new["images"] = list(reversed(entry["images"]))
    return new


def process(name: str, modality: str) -> tuple[int, str]:
    src = ROOT / "data" / name / "data.json"
    if not src.exists():
        raise FileNotFoundError(src)
    dst_dir = ROOT / "data" / f"{name}_reversed"
    dst_dir.mkdir(exist_ok=True)
    dst = dst_dir / "data.json"

    print(f"[{modality:6s}] reading {src.relative_to(ROOT)}")
    entries = json.loads(src.read_text())

    if modality == "text":
        out = [reverse_text_entry(e) for e in entries]
    elif modality == "vision":
        out = [reverse_vision_entry(e) for e in entries]
    else:
        raise ValueError(modality)

    print(f"[{modality:6s}] writing {dst.relative_to(ROOT)} ({len(out)} entries)")
    dst.write_text(json.dumps(out, ensure_ascii=False))
    return len(out), str(dst.relative_to(ROOT))


def main() -> None:
    summary = []
    for name, modality in SOURCES:
        n, path = process(name, modality)
        summary.append((name, modality, n, path))

    print("\n=== Summary ===")
    for name, modality, n, path in summary:
        print(f"  {modality:6s}  {n:>6d}  {path}")


if __name__ == "__main__":
    main()
