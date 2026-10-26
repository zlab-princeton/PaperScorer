"""Shared fixtures for integration tests.

Resolves paths relative to ``tests/fixtures/golden/`` via ``__file__`` so the
tests work on any cluster that has the submodule checked out — no
``/n/fs/...`` hardcodes.

See ``tests/fixtures/golden/`` for the input PDFs + LaTeX sources + the
production ground-truth files used for diffing paperprep output against.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

FIXTURES_GOLDEN = Path(__file__).resolve().parent.parent / "fixtures" / "golden"
INPUTS_DIR = FIXTURES_GOLDEN / "inputs"
GROUND_TRUTH_DIR = FIXTURES_GOLDEN / "ground_truth"

# Stable test set — same papers used everywhere in this module.
ICLR_IDS = ["iclr_ryxOUTVYDH", "iclr_ryxyCeHtPB", "iclr_ryxgJTEYDr"]
ARXIV_IDS = ["arxiv_2201.00346", "arxiv_2301.08170", "arxiv_2511.13144"]
ALL_PDF_IDS = ICLR_IDS + ARXIV_IDS


def have_mineru() -> bool:
    """Return True if the ``mineru`` CLI is on PATH (gpu extra installed).

    Uses ``shutil.which`` rather than ``mineru --version`` — the latter
    initialises ONNX runtime and can take 60-90s on cold cluster nodes,
    which dwarfs any reasonable test-startup budget.
    """
    return shutil.which("mineru") is not None


@pytest.fixture(scope="module")
def mineru_available():
    """Skip the test module entirely when ``mineru`` is not installed."""
    if not have_mineru():
        pytest.skip("mineru CLI not on PATH; run `uv sync --extra gpu` first")
    return True


@pytest.fixture
def pdf_manifest(tmp_path) -> Path:
    """Build a 6-entry manifest of all-PDF inputs (3 iclr + 3 arxiv)."""
    p = tmp_path / "manifest_pdf.jsonl"
    with open(p, "w") as f:
        for pid in ALL_PDF_IDS:
            row = {
                "id": pid,
                "type": "pdf",
                "path": str(INPUTS_DIR / f"{pid}.pdf"),
            }
            f.write(json.dumps(row) + "\n")
    return p


@pytest.fixture
def latex_manifest(tmp_path) -> Path:
    """Build a 9-entry manifest: 6 PDFs + 3 arxiv LaTeX dirs.

    Extracts each .gz into ``tmp_path/latex_sources/<aid>/`` and references that
    dir in the manifest. The .gz files themselves ship in the fixtures dir;
    extraction happens at test time so the unpacked tree isn't committed.
    """
    import gzip
    import tarfile

    latex_src_dir = tmp_path / "latex_sources"
    latex_src_dir.mkdir()

    p = tmp_path / "manifest_latex.jsonl"
    with open(p, "w") as f:
        # 3 ICLR PDFs
        for pid in ICLR_IDS:
            f.write(json.dumps({"id": pid, "type": "pdf", "path": str(INPUTS_DIR / f"{pid}.pdf")}) + "\n")
        # 3 arxiv PDFs
        for pid in ARXIV_IDS:
            f.write(json.dumps({"id": f"{pid}_pdf", "type": "pdf", "path": str(INPUTS_DIR / f"{pid}.pdf")}) + "\n")
        # 3 arxiv LaTeX dirs (extract .gz on the fly)
        for pid in ARXIV_IDS:
            gz_path = INPUTS_DIR / f"{pid}.gz"
            dest = latex_src_dir / pid
            dest.mkdir()
            try:
                with tarfile.open(str(gz_path)) as tf:
                    tf.extractall(str(dest), filter="data")
            except tarfile.ReadError:
                # Single gzipped .tex (rare)
                with gzip.open(str(gz_path), "rb") as gz:
                    content = gz.read()
                (dest / f"{pid}.tex").write_bytes(content)
            f.write(json.dumps({"id": f"{pid}_latex", "type": "latex_dir", "path": str(dest)}) + "\n")
    return p


def load_state(output_dir: Path) -> list[dict]:
    """Read all StateLog rows from ``<output_dir>/state.jsonl``."""
    return [json.loads(l) for l in (output_dir / "state.jsonl").read_text().splitlines() if l.strip()]


def state_by(rows: list[dict], pid: str, stage: str) -> dict | None:
    """Return the latest StateResult row matching (id, stage), or None."""
    found = None
    for r in rows:
        if r.get("id") == pid and r.get("stage") == stage:
            found = r
    return found


def load_ground_truth_json(pid: str, name: str) -> dict:
    """Load a JSON ground-truth file by paper id and basename."""
    p = GROUND_TRUTH_DIR / pid / f"{name}.json"
    return json.loads(p.read_text())


def load_ground_truth_text(pid: str, name: str) -> str:
    """Load a text ground-truth file by paper id and basename (.md / .txt)."""
    for ext in (".md", ".txt"):
        p = GROUND_TRUTH_DIR / pid / f"{name}{ext}"
        if p.exists():
            return p.read_text()
    raise FileNotFoundError(f"no ground-truth {name}.{{md,txt}} under {pid}")


def run_paperprep_local(manifest_path: Path, output_dir: Path, *, stages: str = "compile,mineru,normalize,filter,export",
                         texlive_bin: str | None = None) -> int:
    """Invoke paperprep in local mode (no --slurm) via subprocess."""
    cmd = [
        "uv", "run", "--extra", "gpu", "--no-sync",
        "paperprep", "run",
        "--input-manifest", str(manifest_path),
        "--output-dir", str(output_dir),
        "--stages", stages,
        "--no-resume",
    ]
    if texlive_bin:
        cmd += ["--texlive-bin", texlive_bin]
    # cwd = repo root so the paperprep CLI resolves configs/templates.yaml
    repo_root = Path(__file__).resolve().parent.parent.parent
    r = subprocess.run(cmd, cwd=str(repo_root))
    return r.returncode
