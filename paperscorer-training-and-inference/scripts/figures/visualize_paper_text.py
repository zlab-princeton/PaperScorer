#!/usr/bin/env python3
"""Render a paper's markdown content as a 2x5 grid by typesetting it to PDF.

Pipeline:
    markdown (entry.human, from first `#` on)
        -> pandoc + xelatex -> letter-size PDF (Roboto font)
        -> pymupdf rasterize first 10 pages
        -> matplotlib subplots (same layout as visualize_paper.py)

Usage:
    uv run python scripts/tmp_latex_dir/visualize_paper_text.py \\
        --submission_id FhBT596F1X \\
        --dataset iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_validation

Outputs:
    tmp_latex_dir/figures/paper_text_<submission_id>.{png,pdf}
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path

import fitz  # pymupdf
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image

mpl.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["Roboto", "Arial", "DejaVu Sans"],
})

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "tmp_latex_dir" / "figures"
DATASET_INFO = ROOT / "data" / "dataset_info.json"
FONT_DIR = Path(__file__).resolve().parent / "fonts"

ROWS, COLS = 1, 4
MAX_PANELS = ROWS * COLS  # 4: first 4 pages only, single row

# Pandoc input format: disable TeX math / raw_tex so $ and \ are escaped,
# not interpreted (the paper text is full of LaTeX-y notation that won't
# compile in vanilla xelatex).
PANDOC_FORMAT = (
    "markdown"
    "-tex_math_dollars"
    "-tex_math_single_backslash"
    "-tex_math_double_backslash"
    "-raw_tex"
    "-raw_html"
    "-latex_macros"
)

LATEX_HEADER = r"""
\usepackage{fontspec}
\setmainfont{Roboto-Regular.ttf}[
  Path = %s/,
  BoldFont = Roboto-Bold.ttf
]
""" % FONT_DIR


def resolve_data_json(dataset_name: str) -> Path:
    info = json.loads(DATASET_INFO.read_text())
    if dataset_name not in info:
        raise KeyError(f"Dataset {dataset_name!r} not registered in {DATASET_INFO}")
    file_name = info[dataset_name]["file_name"]
    candidate = ROOT / "data" / file_name
    if candidate.is_dir():
        candidate = candidate / "data.json"
    if not candidate.exists():
        raise FileNotFoundError(candidate)
    return candidate


def find_entry(data: list[dict], submission_id: str) -> dict:
    for e in data:
        sid = (e.get("_metadata") or {}).get("submission_id")
        if sid == submission_id:
            return e
    raise KeyError(f"submission_id {submission_id!r} not found in dataset")


def extract_markdown(entry: dict) -> str:
    """Return the human-prompt content from the first `#` header onward."""
    human = next(c["value"] for c in entry["conversations"] if c["from"] == "human")
    m = re.search(r"^# ", human, flags=re.MULTILINE)
    return human if m is None else human[m.start():]


def md_to_pdf_pages(md: str, dpi: int = 120) -> list[Image.Image]:
    """Run pandoc to typeset md -> letter-size PDF, rasterize each page to PIL."""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        md_path = td / "doc.md"
        pdf_path = td / "doc.pdf"
        header_path = td / "header.tex"
        md_path.write_text(md)
        header_path.write_text(LATEX_HEADER)

        cmd = [
            "pandoc", "-f", PANDOC_FORMAT, str(md_path), "-o", str(pdf_path),
            "--pdf-engine=xelatex",
            "--include-in-header", str(header_path),
            "-V", "geometry:paperwidth=8.5in",
            "-V", "geometry:paperheight=11in",
            "-V", "geometry:margin=0.6in",
            "-V", "fontsize=14pt",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"pandoc failed (rc={proc.returncode}):\n{proc.stderr[-1500:]}")

        doc = fitz.open(pdf_path)
        try:
            pages: list[Image.Image] = []
            for page in doc:
                pix = page.get_pixmap(dpi=dpi)
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                pages.append(img)
        finally:
            doc.close()
        return pages


def render(entry: dict, output_stem: Path) -> None:
    md = extract_markdown(entry)
    pages = md_to_pdf_pages(md)
    n_total = len(pages)
    pages = pages[:MAX_PANELS]
    n_shown = len(pages)

    sample = pages[0] if pages else None
    cell_aspect = (sample.width / sample.height) if sample else 0.77
    panel_h = 6.5
    fig, axes = plt.subplots(
        ROWS, COLS, figsize=(COLS * panel_h * cell_aspect, ROWS * panel_h),
        gridspec_kw={"wspace": 0, "hspace": 0},
    )
    axes = (axes,) if ROWS * COLS == 1 else axes.flatten()

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    for idx, ax in enumerate(axes):
        if idx < n_shown:
            ax.imshow(pages[idx])
        else:
            ax.axis("off")

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    fig.savefig(png_path, dpi=150, bbox_inches="tight", pad_inches=0)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")
    print(f"Pages shown: {n_shown} of {n_total} typeset")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--submission_id", required=True)
    parser.add_argument("--dataset", required=True,
                        help="Registered text dataset from data/dataset_info.json")
    parser.add_argument("--save_name", default=None,
                        help="Filename stem (no extension). Default: paper_text_<submission_id>")
    args = parser.parse_args()

    if "vision" in args.dataset:
        raise ValueError(f"--dataset must be a text dataset; got vision dataset {args.dataset!r}")

    data_json = resolve_data_json(args.dataset)
    print(f"Loading {data_json}")
    data = json.loads(data_json.read_text())

    entry = find_entry(data, args.submission_id)
    stem = args.save_name or f"paper_text_{args.submission_id}"
    render(entry, OUTPUT_DIR / stem)


if __name__ == "__main__":
    main()
