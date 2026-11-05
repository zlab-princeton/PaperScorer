"""Stage: MinerU output -> normalized markdown + redacted PDF + page_images/.

Per paper:
  1. Load content_list.json
  2. Apply ANONYMOUS-mode filter (strip refs, headers, asides)
  3. Render markdown via dataset.canonical_format
  4. compute_anonymous_whiteout + redact_pdf -> redacted PDF
  5. pdf_to_images (150 DPI) -> page_images/page_<N>.png

Output: output_dir/normalized/<id>/{<id>.md, <id>_content_list.json,
                                     <id>_redacted.pdf, page_images/page_<N>.png}
"""
from __future__ import annotations

import json
from pathlib import Path

from paper_anonymizer.core.filters import FilterMode, filter_content_list
from paper_anonymizer.core.pdf import pdf_to_images, redact_pdf
from paper_anonymizer.core.pdf_filter import compute_anonymous_whiteout
from paper_anonymizer.dataset.canonical_format import render_md_from_content_list
from paper_anonymizer.paperprep.progress import paper_iter
from paper_anonymizer.paperprep.state import StageResult


def _normalize_one(
    paper_id: str,
    pdf_path: Path,
    content_list_path: Path,
    middle_json_path: Path | None,
    output_dir: Path,
    *,
    dpi: int = 150,
) -> StageResult:
    out_dir = output_dir / "normalized" / paper_id
    images_dir = out_dir / "page_images"
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    try:
        with open(content_list_path) as f:
            content_list = json.load(f)
    except Exception as e:
        return StageResult(id=paper_id, stage="normalize", status="failed",
                           error=f"failed to load content_list: {e}")

    # Filter (anonymous mode: keep title+abstract, strip refs+headers+asides)
    try:
        filtered, _meta = filter_content_list(content_list, FilterMode.ANONYMOUS)
    except Exception as e:
        return StageResult(id=paper_id, stage="normalize", status="failed",
                           error=f"filter_content_list failed: {e}")

    # Render markdown
    try:
        md = render_md_from_content_list(filtered)
    except Exception as e:
        return StageResult(id=paper_id, stage="normalize", status="failed",
                           error=f"render_md_from_content_list failed: {e}")

    md_path = out_dir / f"{paper_id}.md"
    md_path.write_text(md, encoding="utf-8")
    filtered_cl_path = out_dir / f"{paper_id}_content_list.json"
    filtered_cl_path.write_text(json.dumps(filtered, indent=2), encoding="utf-8")

    # PDF redaction (best-effort — if middle.json is missing, skip the redaction
    # and fall back to a copy of the original PDF; vision images will still be
    # rendered from the unredacted PDF).
    redacted_pdf = out_dir / f"{paper_id}_redacted.pdf"
    if middle_json_path is not None and Path(middle_json_path).exists():
        try:
            # compute_anonymous_whiteout returns (items_list, pages_to_keep_count: int).
            # pages_to_keep is already the body-pages count, not a list of page indices.
            items_to_whiteout, pages_to_keep = compute_anonymous_whiteout(content_list)
            page_end = pages_to_keep if pages_to_keep else None
            redact_pdf(
                pdf_path=pdf_path,
                middle_json_path=Path(middle_json_path),
                output_path=redacted_pdf,
                items_to_whiteout=items_to_whiteout,
                page_end=page_end,
            )
        except Exception as e:
            # Non-fatal: fall back to unredacted PDF for image rasterization
            print(f"WARNING [{paper_id}] redact_pdf failed ({e}); rendering from original PDF")
            redacted_pdf = pdf_path
    else:
        redacted_pdf = pdf_path

    # Rasterize
    try:
        n_pages = pdf_to_images(redacted_pdf, images_dir, dpi=dpi)
    except Exception as e:
        return StageResult(id=paper_id, stage="normalize", status="failed",
                           error=f"pdf_to_images failed: {e}")

    return StageResult(
        id=paper_id, stage="normalize", status="ok",
        output_path=str(out_dir),
        extra={
            "md_path": str(md_path),
            "content_list_path": str(filtered_cl_path),
            "redacted_pdf_path": str(redacted_pdf),
            "n_page_images": n_pages,
        },
    )


def run_chunk(
    items: list[tuple[str, Path, Path, Path | None]],
    output_dir: Path,
    *,
    dpi: int = 150,
) -> list[StageResult]:
    """Submitit-friendly entry: items are (id, pdf_path, content_list_path, middle_json_path)."""
    output_dir = Path(output_dir)
    results: list[StageResult] = []
    for paper_id, pdf, cl, mj in paper_iter(items, "normalize"):
        results.append(_normalize_one(
            paper_id, Path(pdf), Path(cl),
            Path(mj) if mj else None, output_dir, dpi=dpi,
        ))
    return results
