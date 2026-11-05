"""Stage: PDF -> MinerU OCR (content_list.json + middle.json + page images).

Each paper writes to `output_dir/mineru/<id>/vlm/<id>_content_list.json` etc.
GPU-bound; chunk size kept small in submit.py defaults.
"""
from __future__ import annotations

from pathlib import Path

from paper_anonymizer.core.ocr import (
    find_content_list,
    find_middle_json,
    run_mineru_single,
)
from paper_anonymizer.paperprep.progress import paper_iter
from paper_anonymizer.paperprep.state import StageResult


def _mineru_one(
    paper_id: str,
    pdf_path: Path,
    output_dir: Path,
    *,
    max_pages: int,
    mineru_server_url: str | None = None,
) -> StageResult:
    out_dir = output_dir / "mineru" / paper_id
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        result = run_mineru_single(
            pdf_path, out_dir, max_pages=max_pages, server_url=mineru_server_url,
        )
    except Exception as e:
        return StageResult(id=paper_id, stage="mineru", status="failed",
                           error=f"run_mineru_single raised: {e}")

    if not result.get("success"):
        return StageResult(id=paper_id, stage="mineru", status="failed",
                           error=str(result.get("error") or "mineru failed"))

    content_list = find_content_list(out_dir)
    middle_json = find_middle_json(out_dir)
    if content_list is None:
        return StageResult(id=paper_id, stage="mineru", status="failed",
                           error="content_list.json not found in mineru output")

    return StageResult(
        id=paper_id, stage="mineru", status="ok",
        output_path=str(out_dir),
        extra={
            "content_list_path": str(content_list),
            "middle_json_path": str(middle_json) if middle_json else None,
        },
    )


def run_chunk(
    items: list[tuple[str, Path]],
    output_dir: Path,
    *,
    max_pages: int = 14,
    mineru_server_url: str | None = None,
) -> list[StageResult]:
    """Submitit-friendly entry point: process a chunk of (id, pdf_path) pairs."""
    output_dir = Path(output_dir)
    results: list[StageResult] = []
    for paper_id, pdf_path in paper_iter(items, "mineru"):
        results.append(_mineru_one(
            paper_id, Path(pdf_path), output_dir,
            max_pages=max_pages, mineru_server_url=mineru_server_url,
        ))
    return results
