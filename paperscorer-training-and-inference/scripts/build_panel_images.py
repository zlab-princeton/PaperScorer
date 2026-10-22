#!/usr/bin/env python3
"""Build the panel-image vision dataset from existing page screenshots.

For each paper in the vision labelfix v7 filtered24480 dataset
(train/val/test), take its existing page PNGs (already rendered to
data/images/<sid>/page_N_noreferences_original.png by an earlier
extraction pass) and compose the first 10 of them into a single
2240 x 1148 PNG (5 cols x 2 rows of 448 x 574 px panels).

Output is flat: data/images_panel/<sid>.png (one file per paper,
~24,888 total). Sized so Qwen2.5-VL smart_resize is a no-op at
max_pixels=4014080 (3,280 vision tokens / paper).

Run on srun with many workers; skip-if-exists makes the run resumable.

Usage:
    uv run python scripts/build_panel_images.py --workers 32
    # smoke test:
    uv run python scripts/build_panel_images.py --splits validation --limit 5 --workers 1
"""
from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[1]

TARGET_W, TARGET_H = 2380, 1512  # cell aspect 0.630; ~14% horizontal margin vs ICLR content (0.55)
ROWS, COLS = 2, 5
PANEL_W, PANEL_H = TARGET_W // COLS, TARGET_H // ROWS  # 476 x 756
MAX_PAGES = ROWS * COLS  # 10

# Per-venue inner-cell padding (px).  2-column venues (CVPR/ICCV/ICML/AAAI/...)
# trim flush to the cell edges; we inject extra padding so panels for those
# venues have visible page boundaries comparable to ICLR/COLM (which get
# ~30 px natural letterbox from their narrow trimmed content aspect ~0.55).
# Same values used by the contact-sheet inspector dump_arxiv_venue_samples.py.
VENUE_PAGE_BORDER = {
    "iclr": 0,
    "colm": 0,
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

# Source vision datasets (whose `images` lists already point at per-page PNGs)
# and the per-source SID metadata key + panel output dir
SOURCES = {
    "iclr": {
        "base": "iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_labelfix_v7_filtered_filtered24480",
        "sid_key": "submission_id",
        "panel_dir": ROOT / "data" / "images_panel",
    },
    "arxiv": {
        "base": "arxiv_50_50_21k_vision_wmetadata_filtered24480",
        "sid_key": "arxiv_id",
        "panel_dir": ROOT / "data" / "images_panel_arxiv",
    },
}


def trim_white_margins(im: Image.Image, tol: int = 8) -> Image.Image:
    """Crop solid-white margins from the page.

    Uses ImageChops.difference vs a pure-white background, thresholds
    near-white pixels (anti-aliased text edges) at `tol`, then takes
    the bounding box of the remaining content.
    """
    rgb = im.convert("RGB")
    bg = Image.new("RGB", rgb.size, (255, 255, 255))
    diff = ImageChops.difference(rgb, bg).convert("L")
    if tol > 0:
        diff = diff.point(lambda p: 255 if p > tol else 0)
    bbox = diff.getbbox()
    return rgb.crop(bbox) if bbox else rgb


def fit_into_cell(im: Image.Image, cell_w: int, cell_h: int) -> Image.Image:
    """Resize `im` to fit inside (cell_w, cell_h) preserving aspect, on white."""
    cw, ch = im.size
    scale = min(cell_w / cw, cell_h / ch)
    new_w, new_h = max(1, int(round(cw * scale))), max(1, int(round(ch * scale)))
    thumb = im.resize((new_w, new_h), Image.LANCZOS)
    out = Image.new("RGB", (cell_w, cell_h), "white")
    out.paste(thumb, ((cell_w - new_w) // 2, (cell_h - new_h) // 2))
    return out


def compose_panel_from_images(pages: list["Image.Image"], border: int = 0) -> Image.Image:
    """Trim each PIL page, fit into (PANEL_W-2b)x(PANEL_H-2b), tile with `border` px white gutter."""
    canvas = Image.new("RGB", (TARGET_W, TARGET_H), "white")
    inner_w = max(1, PANEL_W - 2 * border)
    inner_h = max(1, PANEL_H - 2 * border)
    for i, src in enumerate(pages[:MAX_PAGES]):
        trimmed = trim_white_margins(src)
        cell = fit_into_cell(trimmed, inner_w, inner_h)
        row, col = divmod(i, COLS)
        canvas.paste(cell, (col * PANEL_W + border, row * PANEL_H + border))
    return canvas


def compose_panel(page_paths: list[Path], border: int = 0) -> Image.Image:
    """File-path version of compose_panel_from_images (used by bulk builder)."""
    pages: list[Image.Image] = []
    for p in page_paths[:MAX_PAGES]:
        with Image.open(p) as src:
            pages.append(src.convert("RGB").copy())
    return compose_panel_from_images(pages, border=border)


def render_one(args: tuple[str, list[str], Path, int]) -> tuple[str, str]:
    sid, image_rels, dst, border = args
    if dst.exists():
        return sid, "skipped"
    try:
        page_paths = [ROOT / r for r in image_rels[:MAX_PAGES]]
        missing = [p for p in page_paths if not p.exists()]
        if missing:
            return sid, f"FAIL: missing pages: {[str(m.name) for m in missing[:3]]}"
        img = compose_panel(page_paths, border=border)
        tmp = dst.with_suffix(".png.tmp")
        img.save(tmp, "PNG", optimize=True, compress_level=6)
        os.rename(tmp, dst)
        return sid, "ok"
    except Exception as e:  # noqa: BLE001
        return sid, f"FAIL: {type(e).__name__}: {e}"


def venue_for(entry: dict) -> str:
    meta = entry.get("_metadata", {}) or {}
    v = meta.get("venue") or meta.get("pl_venue") or "iclr"
    return str(v).lower()


def collect_tasks(source: str, splits: list[str], limit: int | None) -> list[tuple[str, list[str], Path, int]]:
    src = SOURCES[source]
    base = src["base"]; sid_key = src["sid_key"]; panel_dir = src["panel_dir"]
    tasks: list[tuple[str, list[str], Path, int]] = []
    seen: set[str] = set()
    for split in splits:
        ds_path = ROOT / "data" / f"{base}_{split}" / "data.json"
        entries = json.loads(ds_path.read_text())
        if limit is not None:
            entries = entries[:limit]
        for entry in entries:
            sid = entry["_metadata"][sid_key]
            if sid in seen:
                continue
            seen.add(sid)
            border = VENUE_PAGE_BORDER.get(venue_for(entry), DEFAULT_PAGE_BORDER)
            tasks.append((sid, entry["images"], panel_dir / f"{sid}.png", border))
    return tasks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="iclr", choices=list(SOURCES),
                    help="Which dataset family to build panels for")
    ap.add_argument("--splits", nargs="+", default=["train", "validation", "test"],
                    choices=["train", "validation", "test"])
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap entries per split (debug)")
    args = ap.parse_args()

    panel_dir = SOURCES[args.source]["panel_dir"]
    panel_dir.mkdir(parents=True, exist_ok=True)
    tasks = collect_tasks(args.source, args.splits, args.limit)
    print(f"[{args.source}] {len(tasks)} unique papers across {args.splits}; "
          f"workers={args.workers}; out={panel_dir}")

    fails: list[tuple[str, str]] = []
    started_at = time.time()
    skipped = ok = 0
    pool = None
    if args.workers <= 1:
        results_iter = (render_one(t) for t in tasks)
    else:
        pool = Pool(args.workers)
        results_iter = pool.imap_unordered(render_one, tasks, chunksize=4)

    for i, (sid, status) in enumerate(results_iter, 1):
        if status == "ok":
            ok += 1
        elif status == "skipped":
            skipped += 1
        else:
            fails.append((sid, status))
        if i % 500 == 0 or i == len(tasks):
            elapsed = time.time() - started_at
            rate = i / elapsed if elapsed > 0 else 0
            eta = (len(tasks) - i) / rate if rate > 0 else 0
            print(f"  {i}/{len(tasks)}  ok={ok} skipped={skipped} fail={len(fails)}  "
                  f"rate={rate:.2f}/s  eta={eta/60:.1f}min")

    if pool is not None:
        pool.close()
        pool.join()

    print(f"\nDONE  total={len(tasks)}  ok={ok}  skipped={skipped}  fail={len(fails)}  "
          f"elapsed={(time.time()-started_at)/60:.1f}min")
    if fails:
        print("First 20 failures:")
        for sid, status in fails[:20]:
            print(f"  {sid}: {status}")


if __name__ == "__main__":
    main()
