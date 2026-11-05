#!/usr/bin/env python3
"""Dump a contact-sheet image per arxiv conference for visual inspection.

For each venue in the arxiv panel train set, pick N random papers,
tile their existing panel PNGs into one large composite image
(saved as JPEG to keep file sizes small).

Output: tmp_latex_dir/figures/arxiv_venue_samples/<venue>.jpg
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "arxiv_50_50_21k_vision_wmetadata_filtered24480_panel_train" / "data.json"
PANEL_DIR = ROOT / "data" / "images_panel_arxiv"
OUT_DIR = ROOT / "tmp_latex_dir" / "figures" / "arxiv_venue_samples"

SAMPLES_PER_VENUE = 6
SHEET_COLS, SHEET_ROWS = 3, 2          # 3x2 grid of panels per venue
PANEL_W, PANEL_H = 1190, 756           # half the source 2380x1512 -> still readable
LABEL_H = 56                           # px reserved at top for venue+sample id text
SEED = 17

# Per-venue inter-page white border to compensate for source-PDF margin
# differences. Single-column venues (ICLR/COLM/NeurIPS) have wide built-in
# PDF margins so the panel cells already separate visibly; 2-column venues
# (CVPR/ICCV/ICML/AAAI/AISTATS/EccV) trim to cell edges and need explicit
# padding here so each page reads as its own card.
# Sized so the *effective* horizontal margin per page (border + natural
# letterbox bar from aspect mismatch) is roughly consistent across venues
# (~28 px each side, matching ICLR's natural ~30 px bar in a 476x756 cell).
VENUE_PAGE_BORDER = {
    "iclr": 0,    # natural ~30 px
    "colm": 0,    # natural ~30 px
    "neurips": 12,
    "corl": 18,
    "aistats": 22,
    "eccv": 22,
    "acl": 22,
    "icml": 24,
    "iccv": 24,
    "aaai": 24,
    "cvpr": 26,
}
DEFAULT_PAGE_BORDER = 22

# Inner page-grid dims (5 cols x 2 rows) so we can paint borders around each cell.
PANEL_INNER_COLS, PANEL_INNER_ROWS = 5, 2


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)

    entries = json.loads(DATASET.read_text())
    by_venue: dict[str, list[dict]] = defaultdict(list)
    for e in entries:
        v = (e.get("_metadata") or {}).get("venue") or (e.get("_metadata") or {}).get("pl_venue") or "unknown"
        by_venue[v].append(e)

    for venue, papers in sorted(by_venue.items(), key=lambda kv: -len(kv[1])):
        sampled = random.sample(papers, min(SAMPLES_PER_VENUE, len(papers)))
        sheet_w = PANEL_W * SHEET_COLS
        sheet_h = LABEL_H + PANEL_H * SHEET_ROWS
        sheet = Image.new("RGB", (sheet_w, sheet_h), "white")

        # Title strip
        from PIL import ImageDraw, ImageFont
        try:
            font = ImageFont.truetype(str(ROOT / "scripts/tmp_latex_dir/fonts/Roboto-Bold.ttf"), 36)
        except Exception:
            font = ImageFont.load_default()
        ImageDraw.Draw(sheet).text((24, 12), f"venue: {venue}   |   n_in_train={len(papers)}",
                                    fill="black", font=font)

        border = VENUE_PAGE_BORDER.get(venue, DEFAULT_PAGE_BORDER)
        for i, e in enumerate(sampled):
            sid = e["_metadata"]["arxiv_id"]
            png = PANEL_DIR / f"{sid}.png"
            if not png.exists():
                continue
            with Image.open(png) as src:
                src = src.convert("RGB")
                # Add explicit white border around each of the 5x2 inner pages.
                # Source panel is 2380x1512, cells are 476x756 each.
                if border > 0:
                    src_w, src_h = src.size
                    cell_w = src_w // PANEL_INNER_COLS
                    cell_h = src_h // PANEL_INNER_ROWS
                    inner_w = cell_w - 2 * border
                    inner_h = cell_h - 2 * border
                    rebuilt = Image.new("RGB", (src_w, src_h), "white")
                    for r_ in range(PANEL_INNER_ROWS):
                        for c_ in range(PANEL_INNER_COLS):
                            crop = src.crop((c_ * cell_w, r_ * cell_h,
                                              (c_ + 1) * cell_w, (r_ + 1) * cell_h))
                            shrunk = crop.resize((inner_w, inner_h), Image.LANCZOS)
                            rebuilt.paste(shrunk, (c_ * cell_w + border, r_ * cell_h + border))
                    src = rebuilt
                thumb = src.resize((PANEL_W, PANEL_H), Image.LANCZOS)
            r, c = divmod(i, SHEET_COLS)
            x = c * PANEL_W
            y = LABEL_H + r * PANEL_H
            sheet.paste(thumb, (x, y))

        out = OUT_DIR / f"{venue}.jpg"
        sheet.save(out, "JPEG", quality=70, optimize=True)
        sz = out.stat().st_size / 1024
        print(f"  {venue:<10}  n_train={len(papers):>5}  sampled={len(sampled)}  -> {out.name} ({sz:.0f} KB)")


if __name__ == "__main__":
    main()
