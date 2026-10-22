"""Canonical text + vision content builders for ShareGPT entries.

Mirrors AutoReviewer/lib/prep_hf_dataset_for_analysis.py:build_images_only_content
(line 833) so output structure matches the legacy ICLR/NeurIPS dataset format
the project has used for prior runs (see e.g.
LLaMA-Factory/data/2020_2026_85_5_10_split8_balanced_vision_binary_noreviews_v9_validation).

Vision human-turn shape (the `paper_text` argument to create_sharegpt_conversation):

    # {title}

    ## Abstract
    {abstract}

    <image> <image> <image>

create_sharegpt_conversation will then prepend `{task_prompt}\\n\\n` to produce:

    {task_prompt}

    # {title}

    ## Abstract
    {abstract}

    <image> <image> <image>

Text human-turn shape:

    {task_prompt}

    {paper_text}

with literal `<image>` tokens stripped from paper_text (some papers — e.g. VLM
prompt examples — include them as content, which would confuse the trainer's
image-token alignment).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

_IMAGE_TOKEN_PATTERN = re.compile(r"<image>")


def render_md_from_content_list(
    content_list: list[dict],
    body_pages: int | None = None,
) -> str:
    """Render markdown from a (filtered) content_list.

    Mirrors run_arxiv_pipeline.py:_filter_one's rendering loop. Optionally
    truncates to items with `page_idx < body_pages` so the rendered MD
    matches the per-paper body cut even for the ~1.5% of papers where
    pdf_filter._find_references_cutoff missed the refs heading
    (level=2 refs, or no refs heading at all and the cascade falls back
    to appendix/conclusion).
    """
    md_lines: list[str] = []
    for b in content_list:
        if body_pages is not None and int(b.get("page_idx", 0)) >= int(body_pages):
            break
        t = b.get("type", "")
        if t == "text":
            txt = b.get("text", "") or ""
            lvl = b.get("text_level")
            if lvl:
                md_lines.append("#" * min(int(lvl), 6) + " " + txt.strip())
            else:
                md_lines.append(txt)
        elif t == "equation":
            md_lines.append("$$\n" + (b.get("text", "") or "") + "\n$$")
        elif t == "table":
            md_lines.append(b.get("html", b.get("text", "")) or "")
        elif t == "image":
            p = b.get("img_path") or b.get("image_path", "")
            md_lines.append(f"![]({p})")
        elif t == "list":
            for it in b.get("items", []):
                md_lines.append("- " + (it.get("text", "") or ""))
    return "\n\n".join(line for line in md_lines if (line or "").strip())


def strip_image_tokens(s: str) -> str:
    """Remove literal `<image>` tokens from text content."""
    return _IMAGE_TOKEN_PATTERN.sub("", s)


def numerical_page_sort(paths: list[str]) -> list[str]:
    """Sort `page_<N>.png` paths by integer N, not lexically.

    Lex sort puts page_10.png before page_2.png; that's fine when keeping
    all images, but breaks `imgs[:body_pages]` clipping.
    """
    def _key(p: str) -> int:
        try:
            return int(Path(p).stem.rsplit("_", 1)[1])
        except (ValueError, IndexError):
            return 0
    return sorted(paths, key=_key)


def build_text_paper_text(
    content_list_path: str | Path,
    body_pages: int | None,
) -> str:
    """Read content_list.json, render MD truncated to body_pages, strip <image>."""
    p = Path(content_list_path)
    with open(p) as f:
        content_list = json.load(f)
    md = render_md_from_content_list(content_list, body_pages)
    return strip_image_tokens(md)


def build_vision_paper_text(
    title: str | None,
    abstract: str | None,
    n_images: int,
) -> str:
    """Build the canonical vision paper_text section:

        # {title}

        ## Abstract
        {abstract}

        <image> <image> ...

    Skips title/abstract if missing. <image> tokens are space-separated
    (matching legacy `build_images_only_content`). The caller is expected
    to pass this as `paper_text` to create_sharegpt_conversation, which
    prepends `{task_prompt}\\n\\n`.
    """
    # Strip literal <image> tokens from title/abstract (rare: ~0.001% of
    # papers, e.g. VLM papers whose abstract mentions <image> tokens).
    # Without this, the count of <image> in the human turn exceeds
    # len(images) and breaks downstream image-token alignment.
    parts: list[str] = []
    if title and title.strip():
        parts.append(f"# {strip_image_tokens(title.strip())}")
    if abstract and abstract.strip():
        parts.append(f"## Abstract\n{strip_image_tokens(abstract.strip())}")
    if n_images > 0:
        parts.append(" ".join(["<image>"] * n_images))
    return "\n\n".join(parts)
