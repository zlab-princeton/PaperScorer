"""Stage: arxiv-pipeline filters — body_pages>=6, anonymization_verified, image integrity.

Mirrors the production arxiv gates so paperprep outputs are equivalent to
what `run_arxiv_pipeline.py --stage hf_filter` would emit. Drops papers that
fail any of:
  - body_pages < 6 (cuts ACL Short, AAAI Student Abstracts)
  - anonymization_verified == False (no Anonymous / Under Review marker on page 0)
  - any expected page_image missing or empty
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from paper_anonymizer.core.constants import REFERENCES_PATTERN
from paper_anonymizer.paperprep.progress import paper_iter
from paper_anonymizer.paperprep.state import StageResult


# The same anonymization markers verify_anonymization.py looks for.
_ANON_RE = re.compile(
    r"\banonymous\b|confidential\s+review|under\s+review|submission\s*#|blind\s+submission",
    re.IGNORECASE,
)


def _detect_body_pages(content_list: list[dict]) -> tuple[int | None, str]:
    """Cascade detection mirroring scripts/analyze_body_pages.py:_analyze."""
    # 1. References heading at any level
    for item in content_list:
        if item.get("type") == "text" and item.get("text_level"):
            text = item.get("text") or ""
            if REFERENCES_PATTERN.match(text.strip()):
                return item.get("page_idx"), "references_heading"

    # 2. ref_text walk-back (last block of type text where sub_type=="ref_text").
    # Ignore empty-text ref_text items: MinerU sometimes mis-tags a blank block
    # on page 0 as a citation, which would falsely place references at the start.
    nonempty_ref_pages = [
        i.get("page_idx") for i in content_list
        if i.get("sub_type") == "ref_text" and (i.get("text") or "").strip()
    ]
    if nonempty_ref_pages:
        first_ref_page = nonempty_ref_pages[0]
        # Reject implausibly-early walkback results — real references appear
        # near the end of the paper; an "early" first ref_text is almost
        # certainly a parse artifact. Fall through to conclusion / total-pages.
        if first_ref_page is not None and first_ref_page >= 3:
            return first_ref_page, "ref_text_walkback"

    # 3. Conclusion + 1 buffer
    for item in content_list:
        if item.get("type") == "text" and item.get("text_level"):
            text = (item.get("text") or "").strip().lower()
            if "conclusion" in text:
                return (item.get("page_idx", 0) or 0) + 1, "conclusion_buffer"

    # 4. Total pages fallback
    pages = [i.get("page_idx") for i in content_list if i.get("page_idx") is not None]
    if pages:
        return max(pages) + 1, "total_pages_fallback"
    return None, "no_signal"


def _verify_anonymized(content_list: list[dict]) -> bool:
    """Regex-scan for anonymization markers across:
      * any header/footer item (document-wide) -- some classes render the
        "Under Review" banner via fancyhdr, and lab-style classes (zlab.cls)
        put the corresponding-author block into the page-1 footer;
      * every block on page 0 -- the previous first-10-blocks cap missed
        title pages with banner/badge images that consume the window before
        the author region is reached.
    """
    for item in content_list:
        if item.get("type") in ("header", "footer"):
            if _ANON_RE.search(item.get("text") or ""):
                return True
    for i in content_list:
        if i.get("page_idx") != 0:
            continue
        if _ANON_RE.search(i.get("text") or ""):
            return True
    return False


def _filter_one(
    paper_id: str,
    content_list_path: Path,
    images_dir: Path,
    *,
    min_body_pages: int = 6,
) -> StageResult:
    try:
        with open(content_list_path) as f:
            content_list = json.load(f)
    except Exception as e:
        return StageResult(id=paper_id, stage="filter", status="failed",
                           error=f"failed to load filtered content_list: {e}")

    body_pages, detection_method = _detect_body_pages(content_list)
    if body_pages is None or body_pages < min_body_pages:
        return StageResult(id=paper_id, stage="filter", status="failed",
                           error=f"body_pages={body_pages} < {min_body_pages} ({detection_method})",
                           extra={"body_pages": body_pages, "detection_method": detection_method})

    if not _verify_anonymized(content_list):
        return StageResult(id=paper_id, stage="filter", status="failed",
                           error="anonymization_verified=False (no Anonymous/Under Review marker on page 0)",
                           extra={"body_pages": body_pages})

    # Image integrity check
    if not images_dir.is_dir():
        return StageResult(id=paper_id, stage="filter", status="failed",
                           error=f"images_dir missing: {images_dir}",
                           extra={"body_pages": body_pages})
    expected = sorted(images_dir.glob("page_*.png"), key=lambda p: int(p.stem.rsplit("_", 1)[1]))
    if not expected:
        return StageResult(id=paper_id, stage="filter", status="failed",
                           error="no page_*.png files in images_dir",
                           extra={"body_pages": body_pages})
    for img in expected[:body_pages]:
        try:
            if img.stat().st_size == 0:
                return StageResult(id=paper_id, stage="filter", status="failed",
                                   error=f"empty image: {img.name}",
                                   extra={"body_pages": body_pages})
        except FileNotFoundError:
            return StageResult(id=paper_id, stage="filter", status="failed",
                               error=f"missing image: {img.name}",
                               extra={"body_pages": body_pages})

    return StageResult(
        id=paper_id, stage="filter", status="ok",
        extra={
            "body_pages": body_pages,
            "detection_method": detection_method,
            "n_images_available": len(expected),
        },
    )


def run_chunk(
    items: list[tuple[str, Path, Path]],
    *,
    min_body_pages: int = 6,
) -> list[StageResult]:
    """items: (id, content_list_path, images_dir)."""
    results: list[StageResult] = []
    for paper_id, cl, imgs in paper_iter(items, "filter"):
        results.append(_filter_one(paper_id, Path(cl), Path(imgs),
                                   min_body_pages=min_body_pages))
    return results
