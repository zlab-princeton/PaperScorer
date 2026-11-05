"""Loads per-paper page PNGs at scoring time.

The vision reranker expects a list of PIL images per paper, one per page,
matching the training-time convention ``data/images_arxiv/<arxiv_id>/page_N.png``.
This module decouples the absolute filesystem location from the rest of the
server so the deployment can point at whatever directory ``reconstruction.py``
(in ``paperlens-training-and-inference``) materialized.

Resolution order for the images root:

1. Explicit ``images_root`` arg to ``load_pages``
2. ``PAPERLENS_IMAGES_ROOT`` env var
3. ``${PAPERLENS_DATA_ROOT}/images_arxiv`` (if PAPERLENS_DATA_ROOT is set)

Filesystem layout assumed:

    <images_root>/
        <arxiv_id>/
            page_1.png
            page_2.png
            ...

Reuses ``paperprep.dataset.canonical_format.numerical_page_sort`` if importable
(``page_2 < page_10``); falls back to a local copy otherwise.
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Iterable, Optional


log = logging.getLogger(__name__)


_PAGE_NUM_RE = re.compile(r"page[_-]?(\d+)", re.IGNORECASE)


def _numerical_page_sort(paths: Iterable[Path]) -> list[Path]:
    """Sort by integer extracted from filename; unsortable paths go last."""

    def key(p: Path) -> tuple[int, int, str]:
        m = _PAGE_NUM_RE.search(p.stem)
        if m:
            return (0, int(m.group(1)), p.name)
        return (1, 0, p.name)

    return sorted(paths, key=key)


def resolve_images_root(explicit: Optional[str | Path] = None) -> Path:
    """Resolve the images root using the documented precedence."""
    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    env_images = os.environ.get("PAPERLENS_IMAGES_ROOT")
    if env_images:
        return Path(env_images).expanduser().resolve()
    env_data = os.environ.get("PAPERLENS_DATA_ROOT")
    if env_data:
        return (Path(env_data).expanduser() / "images_arxiv").resolve()
    raise RuntimeError(
        "images_root not configured: pass an explicit value, or set "
        "PAPERLENS_IMAGES_ROOT or PAPERLENS_DATA_ROOT"
    )


def resolve_page_paths(
    arxiv_id: str,
    images_root: Optional[str | Path] = None,
    *,
    max_pages: int = 14,
) -> list[str]:
    """Return numerically-sorted absolute paths to a paper's page PNGs.

    Cheap (no PIL load). Use this in the HTTP-client deployment where the
    upstream ``paperlens serve`` does the actual image read via LF's
    multimodal plugin.

    Returns ``[]`` if the per-arxiv_id directory is missing or empty.
    """
    root = resolve_images_root(images_root)
    paper_dir = root / arxiv_id
    if not paper_dir.is_dir():
        log.warning(f"no image dir for arxiv_id={arxiv_id!r} under {root}")
        return []
    pngs = list(paper_dir.glob("page_*.png"))
    if not pngs:
        log.warning(f"no page_*.png under {paper_dir}")
        return []
    return [str(p) for p in _numerical_page_sort(pngs)[:max_pages]]


def load_pages(
    arxiv_id: str,
    images_root: Optional[str | Path] = None,
    *,
    max_pages: int = 14,
) -> list:
    """Load page PNGs for ``arxiv_id`` to PIL.Image (capped at max_pages).

    Use this when you actually need the pixels locally (in-process vLLM,
    not the HTTP-client path). For the HTTP client, ``resolve_page_paths``
    is what you want.
    """
    paths = resolve_page_paths(arxiv_id, images_root, max_pages=max_pages)
    if not paths:
        return []
    from PIL import Image
    images = []
    for p in paths:
        try:
            img = Image.open(p)
            img.load()
            images.append(img)
        except Exception as e:
            log.warning(f"failed to open {p}: {e}")
    return images
