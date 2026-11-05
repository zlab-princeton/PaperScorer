"""Stage: surviving papers -> ShareGPT text + vision JSONs.

For each paper that passed all upstream stages:
  * text entry  = build_text_paper_text(content_list_path, body_pages)
  * vision entry = build_vision_paper_text(title, abstract, n_images)
                   + numerical_page_sort(page_images)[:body_pages]

Both go through create_sharegpt_conversation with label=None — the gpt turn
is dropped (paperprep is inference-only). Output:

    output_dir/sharegpt/text/data.json
    output_dir/sharegpt/vision/data.json

Title + abstract are extracted from the filtered content_list (first
text_level=1 + first text_level=2 by convention; user may override via
manifest `title` / `abstract` extra fields).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from paper_anonymizer.core.constants import (
    SHAREGPT_SYSTEM_PROMPT,
    SHAREGPT_USER_PROMPT_GENERIC,
)
from paper_anonymizer.dataset.canonical_format import (
    build_text_paper_text,
    build_vision_paper_text,
    numerical_page_sort,
)
from paper_anonymizer.dataset.sharegpt import create_sharegpt_conversation


def _extract_title_abstract(content_list: list[dict], extra: dict) -> tuple[str, str]:
    """Best-effort title + abstract extraction.

    Prefers user-supplied values from the manifest's `extra` dict; falls back
    to first text_level=1 (title) and first text containing 'abstract' marker.
    """
    title = extra.get("title")
    abstract = extra.get("abstract")
    if title and abstract:
        return title, abstract

    if not title:
        for item in content_list:
            if item.get("type") == "text" and item.get("text_level") == 1:
                title = (item.get("text") or "").strip()
                if title:
                    break
        if not title:
            title = ""

    if not abstract:
        # MinerU sometimes tags abstract as text_level=2 with text starting "Abstract"
        for item in content_list:
            if item.get("type") != "text":
                continue
            txt = (item.get("text") or "").strip()
            if txt.lower().startswith("abstract"):
                # Walk forward from the "Abstract" heading collecting body text;
                # STOP at the next heading (text_level set) so we don't accidentally
                # absorb the Introduction. Cap at 6 blocks as a safety net.
                idx = content_list.index(item)
                rest = content_list[idx + 1: idx + 7]
                abstract_parts = []
                for r in rest:
                    if r.get("type") != "text":
                        continue
                    if r.get("text_level"):
                        break
                    abstract_parts.append(r.get("text") or "")
                abstract = "\n\n".join(abstract_parts).strip()
                if abstract:
                    break
        if not abstract:
            abstract = ""

    return title, abstract


def build_entries(
    items: list[dict[str, Any]],
    *,
    task_prompt: str = SHAREGPT_USER_PROMPT_GENERIC,
    system_prompt: str = SHAREGPT_SYSTEM_PROMPT,
) -> tuple[list[dict], list[dict]]:
    """Build (text_entries, vision_entries) for the surviving papers.

    Each item dict contains:
        id, content_list_path, images_dir, body_pages, extra
    """
    text_entries: list[dict] = []
    vision_entries: list[dict] = []

    for it in items:
        paper_id = it["id"]
        cl_path = Path(it["content_list_path"])
        images_dir = Path(it["images_dir"])
        body_pages = int(it["body_pages"])
        extra = it.get("extra") or {}

        try:
            with open(cl_path) as f:
                content_list = json.load(f)
        except Exception as e:
            print(f"WARNING [{paper_id}] export: failed to load content_list: {e}")
            continue

        title, abstract = _extract_title_abstract(content_list, extra)

        # Text entry
        try:
            text_paper = build_text_paper_text(cl_path, body_pages=body_pages)
        except Exception as e:
            print(f"WARNING [{paper_id}] export: build_text_paper_text failed: {e}")
            continue

        text_entries.append(create_sharegpt_conversation(
            paper_text=text_paper,
            label=None,
            system_prompt=system_prompt,
            task_prompt=task_prompt,
            metadata={
                "id": paper_id,
                "body_pages": body_pages,
                **{k: v for k, v in extra.items() if k not in ("title", "abstract")},
            },
        ))

        # Vision entry
        try:
            page_paths_all = list(images_dir.glob("page_*.png"))
            page_paths = numerical_page_sort(page_paths_all)[:body_pages]
        except Exception as e:
            print(f"WARNING [{paper_id}] export: vision page glob failed: {e}")
            continue
        vision_paper = build_vision_paper_text(title, abstract, len(page_paths))
        vision_entries.append(create_sharegpt_conversation(
            paper_text=vision_paper,
            label=None,
            system_prompt=system_prompt,
            task_prompt=task_prompt,
            metadata={
                "id": paper_id,
                "body_pages": body_pages,
                "n_images": len(page_paths),
                **{k: v for k, v in extra.items() if k not in ("title", "abstract")},
            },
            images=[str(p) for p in page_paths],
        ))

    return text_entries, vision_entries


def write_outputs(
    text_entries: list[dict],
    vision_entries: list[dict],
    output_dir: Path,
) -> tuple[Path, Path]:
    text_dir = output_dir / "sharegpt" / "text"
    vision_dir = output_dir / "sharegpt" / "vision"
    text_dir.mkdir(parents=True, exist_ok=True)
    vision_dir.mkdir(parents=True, exist_ok=True)
    text_path = text_dir / "data.json"
    vision_path = vision_dir / "data.json"
    with open(text_path, "w") as f:
        json.dump(text_entries, f, indent=2)
    with open(vision_path, "w") as f:
        json.dump(vision_entries, f, indent=2)
    return text_path, vision_path
