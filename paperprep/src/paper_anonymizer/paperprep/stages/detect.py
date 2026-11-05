"""Stage: page-count gate.

The user's plan is "drop pdf where page_count < 4" before MinerU. This is a
fast local check; not worth a full submitit array. Run inline on the driver
after the compile stage.
"""
from __future__ import annotations

from pathlib import Path

from paper_anonymizer.core.latex_compile import get_pdf_page_count
from paper_anonymizer.paperprep.state import StageResult


MIN_PAGES = 4


def filter_short_pdfs(
    pdf_paths: list[tuple[str, Path]],
    *,
    min_pages: int = MIN_PAGES,
) -> tuple[list[tuple[str, Path]], list[StageResult]]:
    """Split (id, pdf_path) pairs into (kept, dropped_results)."""
    kept: list[tuple[str, Path]] = []
    dropped: list[StageResult] = []
    for paper_id, pdf in pdf_paths:
        pc = get_pdf_page_count(pdf)
        if pc is None:
            dropped.append(StageResult(
                id=paper_id, stage="page_count_filter", status="failed",
                error="pdfinfo could not read page count"))
            continue
        if pc < min_pages:
            dropped.append(StageResult(
                id=paper_id, stage="page_count_filter", status="failed",
                error=f"page_count={pc} < {min_pages}",
                extra={"page_count": pc}))
            continue
        kept.append((paper_id, pdf))
    return kept, dropped
