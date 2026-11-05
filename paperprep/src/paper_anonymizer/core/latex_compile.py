"""LaTeX source extraction + pdflatex compilation helpers.

Given a `.gz` / `.tar` / `.tar.gz` arXiv source bundle, extracts it into a
temporary directory, finds the main `.tex` file, and runs `pdflatex` with a
timeout. Engine-agnostic source operations (find main tex, parse README) are
separated from the actual compile so they stay testable without needing TeX.
"""
from __future__ import annotations

import gzip
import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path


def extract_source(gz_path: Path, dest_dir: Path) -> bool:
    """Extract an arXiv source bundle into ``dest_dir``.

    arXiv source bundles come in three shapes:
      - a gzipped tar (multi-file project)
      - a gzipped single ``.tex`` file
      - (rare) a plain tar

    Returns ``True`` on success. Creates ``dest_dir`` if needed.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)

    # First try as tar (handles .tar.gz and .tar transparently).
    try:
        with tarfile.open(str(gz_path)) as tf:
            tf.extractall(str(dest_dir), filter="data")
        return True
    except tarfile.ReadError:
        pass
    except Exception:
        return False

    # Fall back: single gzipped .tex file.
    try:
        with gzip.open(str(gz_path), "rb") as gz:
            content = gz.read()
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            text = content.decode("latin-1", errors="replace")
        (dest_dir / f"{gz_path.stem}.tex").write_text(text, encoding="utf-8")
        return True
    except Exception:
        return False


def find_main_tex(src_dir: Path) -> Path | None:
    """Locate the main `.tex` file in an extracted arXiv source tree.

    Strategy (first hit wins):
      1. If ``00README.json`` exists and names a ``toplevelfile``, use that.
      2. Otherwise scan all ``.tex`` files; the first one containing
         ``\\documentclass`` wins (shallowest path breaks ties).
    """
    readme = src_dir / "00README.json"
    if readme.is_file():
        try:
            data = json.loads(readme.read_text(encoding="utf-8", errors="replace"))
            sources = data.get("sources") or []
            for entry in sources:
                top = entry.get("filename") or entry.get("toplevelfile")
                if top and top.endswith(".tex"):
                    cand = src_dir / top
                    if cand.is_file():
                        return cand
        except Exception:
            pass

    tex_files = sorted(src_dir.rglob("*.tex"), key=lambda p: (len(p.parts), str(p)))
    for tex_file in tex_files:
        try:
            head = tex_file.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        if "\\documentclass" in head:
            return tex_file
    return None


def run_pdflatex(
    main_tex: Path,
    *,
    passes: int = 3,
    timeout: int = 300,
    texlive_bin: str | None = None,
    run_bibtex: bool = True,
) -> dict:
    """Run ``pdflatex`` on ``main_tex`` with a timeout.

    Fallback compiler for nodes without ``latexmk``. Sequence:
    pdflatex (pass 1) -> bibtex (best-effort, if run_bibtex) -> pdflatex x(passes-1)
    so both the bibliography and cross-references resolve. Returns:

        {
            'success': bool,
            'pdf_path': str | None,
            'returncode': int,
            'timed_out': bool,
            'passes_run': int,
            'error': str | None,
        }
    """
    env = os.environ.copy()
    if texlive_bin:
        env["PATH"] = f"{texlive_bin}:{env.get('PATH', '')}"

    workdir = main_tex.parent
    stem = main_tex.stem
    pdf = workdir / f"{stem}.pdf"

    result = {
        "success": False,
        "pdf_path": None,
        "returncode": -1,
        "timed_out": False,
        "passes_run": 0,
        "error": None,
    }

    # No -halt-on-error: let pdflatex power through non-fatal errors
    # (missing optional packages, undefined commands) and still produce
    # a PDF. Many LaTeX errors are recoverable.
    cmd = [
        "pdflatex",
        "-interaction=nonstopmode",
        "-file-line-error",
        main_tex.name,
    ]

    for pass_i in range(passes):
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(workdir),
                env=env,
                capture_output=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            result["timed_out"] = True
            result["error"] = f"pdflatex timed out after {timeout}s on pass {pass_i + 1}"
            # Even on timeout, a partial PDF may exist
            if pdf.exists():
                result["success"] = True
                result["pdf_path"] = str(pdf)
            return result
        except FileNotFoundError:
            result["error"] = "pdflatex not found on PATH"
            return result

        result["returncode"] = proc.returncode
        result["passes_run"] = pass_i + 1

        # After the first pass the .aux exists; run bibtex once so \cite
        # commands resolve in the remaining passes. Best-effort: a paper with
        # no \bibliography just yields a bibtex error we ignore.
        if run_bibtex and pass_i == 0:
            try:
                subprocess.run(
                    ["bibtex", stem],
                    cwd=str(workdir), env=env,
                    capture_output=True, timeout=timeout,
                )
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass  # no bibtex / timed out -> proceed; refs may show as [?]

    # Check for PDF regardless of return code — nonstopmode often
    # produces a valid PDF even with rc=1.
    if pdf.exists():
        result["success"] = True
        result["pdf_path"] = str(pdf)
    else:
        result["error"] = f"no PDF produced (rc={result['returncode']})"
    return result


def run_latexmk(
    main_tex: Path,
    *,
    timeout: int = 600,
    texlive_bin: str | None = None,
) -> dict:
    """Run ``latexmk -pdf`` on ``main_tex`` — handles pdflatex + bibtex +
    rerun passes automatically. Returns the same shape as ``run_pdflatex``."""
    env = os.environ.copy()
    if texlive_bin:
        env["PATH"] = f"{texlive_bin}:{env.get('PATH', '')}"

    workdir = main_tex.parent
    stem = main_tex.stem
    pdf = workdir / f"{stem}.pdf"

    result = {
        "success": False,
        "pdf_path": None,
        "returncode": -1,
        "timed_out": False,
        "passes_run": 0,
        "error": None,
    }

    cmd = [
        "latexmk",
        "-pdf",
        "-interaction=nonstopmode",
        "-file-line-error",
        "-f",  # force-mode: keep going on recoverable errors
        main_tex.name,
    ]

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(workdir),
            env=env,
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        result["timed_out"] = True
        result["error"] = f"latexmk timed out after {timeout}s"
        if pdf.exists():
            result["success"] = True
            result["pdf_path"] = str(pdf)
        return result
    except FileNotFoundError:
        result["error"] = "latexmk not found on PATH"
        return result

    result["returncode"] = proc.returncode

    if pdf.exists():
        result["success"] = True
        result["pdf_path"] = str(pdf)
    else:
        result["error"] = f"no PDF produced (rc={result['returncode']})"
    return result


def get_pdf_page_count(pdf_path: Path) -> int | None:
    """Return the number of pages in a PDF, or ``None`` on failure.
    Prefers ``pdfinfo`` (poppler-utils) when on PATH; falls back to pymupdf
    so deployments without poppler still work (pymupdf is already a hard dep)."""
    try:
        proc = subprocess.run(
            ["pdfinfo", str(pdf_path)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if proc.returncode == 0:
            for line in proc.stdout.splitlines():
                if line.startswith("Pages:"):
                    try:
                        return int(line.split(":", 1)[1].strip())
                    except ValueError:
                        break
    except FileNotFoundError:
        pass
    except Exception:
        return None
    try:
        import pymupdf  # type: ignore
        with pymupdf.open(str(pdf_path)) as doc:
            return doc.page_count
    except Exception:
        return None


def cleanup_source_dir(src_dir: Path) -> None:
    """Remove a staging source directory, ignoring errors."""
    try:
        shutil.rmtree(src_dir, ignore_errors=True)
    except Exception:
        pass
