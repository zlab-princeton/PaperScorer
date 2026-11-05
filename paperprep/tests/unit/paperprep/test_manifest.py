"""Tests for paperprep.manifest — input parsing + validation."""
import json
from pathlib import Path

import pytest

from paper_anonymizer.paperprep.manifest import (
    PaperEntry,
    build_manifest_from_dir,
    load_manifest,
    write_manifest,
)


@pytest.fixture
def latex_dir(tmp_path: Path) -> Path:
    d = tmp_path / "latex_papers"
    (d / "paper001").mkdir(parents=True)
    (d / "paper001" / "main.tex").write_text("\\documentclass{article}")
    (d / "paper002").mkdir(parents=True)
    (d / "paper002" / "00README.json").write_text('{"sources": []}')
    return d


@pytest.fixture
def pdf_dir(tmp_path: Path) -> Path:
    d = tmp_path / "pdfs"
    d.mkdir()
    (d / "a.pdf").write_bytes(b"%PDF-1.4\n")
    (d / "b.pdf").write_bytes(b"%PDF-1.4\n")
    return d


def test_build_from_latex_dir(latex_dir: Path):
    entries = build_manifest_from_dir(latex_dir, "latex_dir")
    assert len(entries) == 2
    assert {e.id for e in entries} == {"paper001", "paper002"}
    for e in entries:
        assert e.type == "latex_dir"
        assert e.path.is_dir()


def test_build_from_pdf_dir(pdf_dir: Path):
    entries = build_manifest_from_dir(pdf_dir, "pdf")
    assert len(entries) == 2
    assert {e.id for e in entries} == {"a", "b"}
    for e in entries:
        assert e.type == "pdf"
        assert e.path.is_file()


def test_load_manifest_jsonl(tmp_path: Path, latex_dir: Path):
    p = tmp_path / "manifest.jsonl"
    p.write_text(
        json.dumps({"id": "p1", "type": "latex_dir", "path": str(latex_dir / "paper001")}) + "\n"
        + json.dumps({"id": "p2", "type": "latex_dir", "path": str(latex_dir / "paper002"), "main_tex": "main.tex"}) + "\n"
    )
    entries = load_manifest(p)
    assert len(entries) == 2
    assert entries[1].main_tex == "main.tex"


def test_load_manifest_passes_extra_fields(tmp_path: Path, pdf_dir: Path):
    p = tmp_path / "manifest.jsonl"
    p.write_text(json.dumps({
        "id": "x", "type": "pdf", "path": str(pdf_dir / "a.pdf"),
        "title": "Cool Paper", "abstract": "abstract here",
    }) + "\n")
    [entry] = load_manifest(p)
    assert entry.extra == {"title": "Cool Paper", "abstract": "abstract here"}


def test_load_manifest_rejects_duplicate_ids(tmp_path: Path, pdf_dir: Path):
    p = tmp_path / "manifest.jsonl"
    p.write_text(
        json.dumps({"id": "x", "type": "pdf", "path": str(pdf_dir / "a.pdf")}) + "\n"
        + json.dumps({"id": "x", "type": "pdf", "path": str(pdf_dir / "b.pdf")}) + "\n"
    )
    with pytest.raises(ValueError, match="duplicate id"):
        load_manifest(p)


def test_load_manifest_rejects_missing_path(tmp_path: Path):
    p = tmp_path / "manifest.jsonl"
    p.write_text(json.dumps({"id": "x", "type": "pdf", "path": "/nonexistent/x.pdf"}) + "\n")
    with pytest.raises(ValueError, match="does not exist"):
        load_manifest(p)


def test_load_manifest_rejects_empty(tmp_path: Path):
    p = tmp_path / "manifest.jsonl"
    p.write_text("")
    with pytest.raises(ValueError, match="empty"):
        load_manifest(p)


def test_write_then_load_roundtrip(tmp_path: Path, pdf_dir: Path):
    entries = [
        PaperEntry(id="r1", type="pdf", path=pdf_dir / "a.pdf", extra={"k": "v"}),
        PaperEntry(id="r2", type="pdf", path=pdf_dir / "b.pdf"),
    ]
    out = tmp_path / "round.jsonl"
    write_manifest(entries, out)
    parsed = load_manifest(out)
    assert {e.id for e in parsed} == {"r1", "r2"}
    assert parsed[0].extra == {"k": "v"}
