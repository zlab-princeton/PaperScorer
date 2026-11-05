"""Compose a paper's first N rendered pages into a single 5x2 grid PNG.

Used by the trajectory drilldown preview: each commit's compiled paper
gets a single panel image, which the UI swaps in when you click a point
on the line graph. Best-effort -- if Pillow isn't installed or any page
fails to open, returns None and scoring proceeds normally.

Slim version of ``paperlens-training-and-inference/scripts/build_panel_images.py``
(no per-venue padding, no multiprocessing -- this runs once per scoring call
inline, so the panel sits next to ``data.json`` in the paperprep output dir).
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional


log = logging.getLogger(__name__)

ROWS, COLS = 2, 5
MAX_PAGES = ROWS * COLS  # 10

# Slightly wider than tall; cell aspect close to ICLR letter pages without
# excessive horizontal padding. 5 cols x 320 w; 2 rows x 512 h.
TARGET_W, TARGET_H = 1600, 1024
PANEL_W = TARGET_W // COLS
PANEL_H = TARGET_H // ROWS
BORDER = 4  # px of white gutter between cells


def _page_num(p: Path) -> int:
    """Sort key for ``page_N.png`` / ``page_N_*.png`` files (numerical not lex)."""
    m = re.search(r"page_(\d+)", p.stem)
    return int(m.group(1)) if m else 0


def build_panel(page_pngs: list[Path], dest: Path) -> Optional[Path]:
    """Build a 5x2 grid from the first 10 of ``page_pngs`` (sorted by page
    index) and write to ``dest``. Returns the path on success, ``None`` on any
    error -- panel preview is a UI nicety, never block scoring on it.
    """
    try:
        from PIL import Image
    except ImportError:
        return None
    pngs = sorted([Path(p) for p in page_pngs if Path(p).is_file()], key=_page_num)[:MAX_PAGES]
    if not pngs:
        return None
    try:
        canvas = Image.new("RGB", (TARGET_W, TARGET_H), "white")
        cw, ch = PANEL_W - 2 * BORDER, PANEL_H - 2 * BORDER
        for i, p in enumerate(pngs):
            try:
                img = Image.open(p).convert("RGB")
            except Exception:
                continue
            img.thumbnail((cw, ch), Image.LANCZOS)
            row, col = divmod(i, COLS)
            x = col * PANEL_W + BORDER + (cw - img.width) // 2
            y = row * PANEL_H + BORDER + (ch - img.height) // 2
            canvas.paste(img, (x, y))
        dest.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(dest, format="PNG", optimize=True)
        return dest
    except Exception as e:
        log.warning("panel build failed (%s): %s", dest, e)
        return None
