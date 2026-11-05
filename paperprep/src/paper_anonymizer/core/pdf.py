"""PDF redaction and page image generation.

Ported from:
    - lib/normalize_mineru.py:process_pdf(), normalized_to_pdf_points(), load_page_sizes_from_middle_json()
    - lib/pdf_to_images.py:pdf_to_images()
"""

import json
from collections import defaultdict
from pathlib import Path

import fitz  # PyMuPDF


def load_page_sizes(middle_json_path: Path) -> dict[int, tuple[float, float]]:
    """Load page sizes from MinerU middle.json for coordinate conversion."""
    try:
        with open(middle_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return {i: tuple(page['page_size']) for i, page in enumerate(data['pdf_info'])}
    except Exception:
        return {}


def normalized_to_pdf_points(bbox: list, page_size: tuple) -> list:
    """Convert normalized 0-1000 bbox to PDF points.

    MinerU content_list bboxes use normalized 0-1000 range:
        normalized = (pdf_points * 1000) / page_dimension
        pdf_points = (normalized * page_dimension) / 1000
    """
    x0, y0, x1, y1 = bbox
    page_width, page_height = page_size
    return [
        x0 * page_width / 1000,
        y0 * page_height / 1000,
        x1 * page_width / 1000,
        y1 * page_height / 1000,
    ]


def redact_pdf(
    pdf_path: Path,
    middle_json_path: Path,
    output_path: Path,
    items_to_whiteout: list[dict],
    page_end: int,
) -> bool:
    """Redact regions on a PDF and trim pages after a cutoff.

    Args:
        pdf_path: Path to the original PDF.
        middle_json_path: Path to MinerU middle.json (for coordinate conversion).
        output_path: Where to save the redacted PDF.
        items_to_whiteout: List of dicts with 'bbox' and 'page_idx' keys.
        page_end: Last page to keep (0-indexed, inclusive).

    Returns:
        True on success, False on error.
    """
    doc = None
    try:
        page_sizes = load_page_sizes(middle_json_path)

        redact_by_page = defaultdict(list)
        for item in items_to_whiteout:
            bbox = item.get('bbox')
            page_idx = item.get('page_idx', 0)
            if bbox and len(bbox) >= 4:
                redact_by_page[page_idx].append(bbox)

        doc = fitz.open(str(pdf_path))
        max_page = min(len(doc), page_end + 1)

        for page_idx in range(max_page):
            if page_idx not in redact_by_page:
                continue

            page = doc[page_idx]
            page_rect = page.rect
            page_size = page_sizes.get(page_idx, (page_rect.width, page_rect.height))

            for bbox in redact_by_page[page_idx]:
                x0, y0, x1, y1 = normalized_to_pdf_points(bbox, page_size)
                rect = fitz.Rect(x0, y0, x1, y1)
                page.add_redact_annot(rect, fill=(1, 1, 1))

            try:
                page.apply_redactions()
            except Exception as e:
                print(f"Warning: redaction failed on page {page_idx} of {pdf_path}: {e}")

        if len(doc) > page_end + 1:
            doc.delete_pages(from_page=page_end + 1, to_page=len(doc) - 1)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        doc.save(str(output_path), garbage=4, deflate=True)
        doc.close()
        doc = None
        return True

    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {e}")
        return False
    finally:
        if doc is not None:
            try:
                doc.close()
            except Exception:
                pass


def pdf_to_images(pdf_path: Path, output_dir: Path, dpi: int = 150) -> int:
    """Convert a PDF to per-page PNG images.

    Args:
        pdf_path: Path to the PDF file.
        output_dir: Directory to save page_N.png files.
        dpi: Resolution (default 150).

    Returns:
        Number of pages converted.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    doc = fitz.open(str(pdf_path))
    num_pages = len(doc)

    for i, page in enumerate(doc):
        pix = page.get_pixmap(matrix=matrix)
        output_path = output_dir / f"page_{i + 1}.png"
        pix.save(str(output_path))

    doc.close()
    return num_pages
