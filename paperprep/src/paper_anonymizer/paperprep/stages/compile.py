"""Stage: LaTeX source dir -> compiled (anonymized) PDF.

Per paper of `type=latex_dir`:
  1. Copy src dir to a tempdir
  2. Apply venue toggle (anonymize \\author / [final] / [accepted] / etc.)
  3. Scrub TeX (remove ack / repro / leaky footnotes / URLs)
  4. Detect template venue (warn if not detected)
  5. Run latexmk; copy resulting PDF to output_dir/compiled/<id>.pdf

Skip type=pdf entries — they're forwarded as-is to the next stage.

Returns one StageResult per paper.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from paper_anonymizer.core.latex_compile import (
    extract_source,
    find_main_tex,
    get_pdf_page_count,
    run_latexmk,
    run_pdflatex,
)
from paper_anonymizer.core.tex import scrub_tex_dir
from paper_anonymizer.core.venue_toggle import apply_venue_toggle
from paper_anonymizer.paperprep.manifest import PaperEntry
from paper_anonymizer.paperprep.progress import paper_iter
from paper_anonymizer.paperprep.state import StageResult


def _resolve_main_tex(src_dir: Path, hint: str | None) -> Path | None:
    """If user gave main_tex, use it; else fall back to find_main_tex."""
    if hint:
        cand = src_dir / hint
        if cand.is_file():
            return cand
    return find_main_tex(src_dir)


def _detect_template_venue(staging: Path) -> tuple[str | None, int | None]:
    """Run sty/template detection across all .tex in staging dir.

    Returns (venue, year) or (None, None) if no template matched. Reuses the
    arxiv pipeline's existing scan helpers via a local import (so paperprep
    can be imported on systems without rescan_gz_templates available).
    """
    try:
        # scripts/ is on sys.path because it's installed as part of the project? No —
        # it isn't a package. We re-implement the minimal scan here using the
        # same configs/templates.yaml + regex pattern shape.
        import re
        import yaml

        # Find configs/templates.yaml relative to repo root or import-time location.
        candidates = [
            Path(__file__).resolve().parents[4] / "configs" / "templates.yaml",
            Path("configs/templates.yaml"),
        ]
        cfg_path = next((p for p in candidates if p.exists()), None)
        if cfg_path is None:
            return None, None

        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)

        templates = {}
        for venue_key, venue_config in cfg.get("templates", {}).items():
            patterns = [re.compile(p) for p in venue_config.get("patterns", [])]
            year_re = re.compile(venue_config["year_regex"]) if venue_config.get("year_regex") else None
            templates[venue_key] = (patterns, year_re)

        for tex_file in staging.rglob("*.tex"):
            try:
                content = tex_file.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            for line in content.split("\n"):
                if line.lstrip().startswith("%"):
                    continue
                for venue_key, (patterns, year_re) in templates.items():
                    for pat in patterns:
                        if pat.search(line):
                            year = None
                            if year_re:
                                m = year_re.search(line)
                                if m:
                                    try:
                                        ys = m.group(1)
                                        y = int(ys)
                                        if len(ys) == 2:
                                            y = 2000 + y if y < 50 else 1900 + y
                                        year = y
                                    except (ValueError, IndexError):
                                        pass
                            return venue_key, year
        return None, None
    except Exception:
        return None, None


def _compile_one(entry: PaperEntry, output_dir: Path, *, timeout: int, texlive_bin: str | None) -> StageResult:
    """Process one latex_dir entry. PDF entries pass through (status=skipped)."""
    if entry.type == "pdf":
        # PDFs skip compile and feed straight into the page-count filter
        return StageResult(
            id=entry.id,
            stage="compile",
            status="skipped",
            output_path=str(entry.path),
            extra={"reason": "input is already a pdf", "page_count": _try_page_count(entry.path)},
        )

    pdf_dest = output_dir / "compiled" / f"{entry.id}.pdf"
    pdf_dest.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f"paperprep_compile_{entry.id}_") as tmp:
        staging = Path(tmp)
        try:
            shutil.copytree(entry.path, staging, dirs_exist_ok=True)
        except Exception as e:
            return StageResult(id=entry.id, stage="compile", status="failed",
                               error=f"copytree failed: {e}")

        # Anonymize + scrub all .tex files in-place
        toggle_changes = 0
        for tex_file in staging.rglob("*.tex"):
            try:
                orig = tex_file.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            new, n = apply_venue_toggle(orig)
            if n > 0 and new != orig:
                tex_file.write_text(new, encoding="utf-8")
                toggle_changes += n
        try:
            scrub_stats = scrub_tex_dir(staging)
        except Exception as e:
            return StageResult(id=entry.id, stage="compile", status="failed",
                               error=f"scrub_tex_dir failed: {e}")

        # Template detection (warn-only)
        venue, year = _detect_template_venue(staging)
        warn = None if venue else "no LaTeX template venue detected (check configs/templates.yaml)"
        if warn:
            print(f"WARNING [{entry.id}]: {warn}")

        # Find main tex
        main_tex = _resolve_main_tex(staging, entry.main_tex)
        if main_tex is None:
            return StageResult(id=entry.id, stage="compile", status="failed",
                               error="no main .tex found (00README.json missing or no \\documentclass)")

        # Compile. Prefer latexmk (handles bibtex + reruns); fall back to a
        # pdflatex+bibtex sequence on nodes without latexmk installed.
        result = run_latexmk(main_tex, timeout=timeout, texlive_bin=texlive_bin)
        if not result.get("success") and "latexmk not found" in (result.get("error") or ""):
            print(f"INFO [{entry.id}]: latexmk unavailable, falling back to pdflatex+bibtex")
            result = run_pdflatex(main_tex, passes=3, timeout=timeout, texlive_bin=texlive_bin)
        pdf_built = result.get("pdf_path")
        if not result.get("success") or not pdf_built or not Path(pdf_built).exists():
            return StageResult(id=entry.id, stage="compile", status="failed",
                               error=str(result.get("error") or "compile produced no pdf"))

        try:
            shutil.copyfile(pdf_built, pdf_dest)
        except Exception as e:
            return StageResult(id=entry.id, stage="compile", status="failed",
                               error=f"pdf copy failed: {e}")

        page_count = get_pdf_page_count(pdf_dest)
        return StageResult(
            id=entry.id,
            stage="compile",
            status="ok",
            output_path=str(pdf_dest),
            extra={
                "page_count": page_count,
                "toggle_changes": toggle_changes,
                "scrub_stats": scrub_stats,
                "template_venue": venue,
                "template_year": year,
                "warning": warn,
            },
        )


def _try_page_count(pdf_path: Path) -> int | None:
    try:
        return get_pdf_page_count(pdf_path)
    except Exception:
        return None


def run_chunk(
    entries: list[PaperEntry],
    output_dir: Path,
    *,
    timeout: int = 600,
    texlive_bin: str | None = None,
) -> list[StageResult]:
    """Submitit-friendly entry point: process a chunk of entries serially."""
    output_dir = Path(output_dir)
    results: list[StageResult] = []
    for entry in paper_iter(entries, "compile"):
        results.append(_compile_one(entry, output_dir, timeout=timeout, texlive_bin=texlive_bin))
    return results
