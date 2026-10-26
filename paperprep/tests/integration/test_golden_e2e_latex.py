"""Ground-truth correctness test — LaTeX dir + PDF inputs (full pipeline).

Adds 3 ``type=latex_dir`` entries to the manifest (same 3 arxiv papers, just
extracted from their .gz sources), so the compile stage actually runs.
Exercises:

  - `extract_source` / `find_main_tex` / `apply_venue_toggle` / `scrub_tex_dir`
  - `run_latexmk` end-to-end on real arxiv sources (incl. Meta's custom
    `fairmeta.cls` if 2503.10622 is in the set)
  - Sty/template detection against `configs/templates.yaml`
  - All downstream stages on a paperprep-compiled (not production-compiled) PDF

Tagged ``slow`` because it adds ~3 latex compiles + ~3 extra MinerU passes
on top of the pdf-only test. Wall time: ~40 min on an L40.

    uv run pytest tests/integration/test_golden_e2e_latex.py -m slow --no-header -s
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.integration.conftest import (
    ARXIV_IDS,
    GROUND_TRUTH_DIR,
    ICLR_IDS,
    INPUTS_DIR,
    load_state,
    run_paperprep_local,
    state_by,
)


PDF_PAGE_COUNT_TOLERANCE = 1   # paperprep-compiled PDF can differ ±1 page vs prod


pytestmark = [pytest.mark.slow, pytest.mark.gpu]


@pytest.fixture(scope="module")
def latex_manifest_module(tmp_path_factory):
    """Build the 9-entry manifest: 3 ICLR PDF + 3 arxiv PDF + 3 arxiv LaTeX dir."""
    import gzip
    import tarfile
    tp = tmp_path_factory.mktemp("manifest_latex")
    latex_src_dir = tp / "latex_sources"
    latex_src_dir.mkdir()
    p = tp / "manifest.jsonl"
    with open(p, "w") as f:
        for pid in ICLR_IDS:
            f.write(json.dumps({"id": pid, "type": "pdf", "path": str(INPUTS_DIR / f"{pid}.pdf")}) + "\n")
        for pid in ARXIV_IDS:
            f.write(json.dumps({"id": f"{pid}_pdf", "type": "pdf", "path": str(INPUTS_DIR / f"{pid}.pdf")}) + "\n")
        for pid in ARXIV_IDS:
            gz = INPUTS_DIR / f"{pid}.gz"
            dest = latex_src_dir / pid
            dest.mkdir()
            try:
                with tarfile.open(str(gz)) as tf:
                    tf.extractall(str(dest), filter="data")
            except tarfile.ReadError:
                with gzip.open(str(gz), "rb") as gzf:
                    (dest / f"{pid}.tex").write_bytes(gzf.read())
            f.write(json.dumps({"id": f"{pid}_latex", "type": "latex_dir", "path": str(dest)}) + "\n")
    return p


@pytest.fixture(scope="module")
def latex_run_output(tmp_path_factory, mineru_available, latex_manifest_module):
    """Run paperprep on the 9-entry manifest exactly once, share across tests.

    Requires TeX Live on PATH (compile stage on the latex_dir entries). Uses
    the well-known cluster install path; override via PAPERPREP_TEXLIVE_BIN.
    """
    import os
    out = tmp_path_factory.mktemp("latex_run_out")
    texlive = os.environ.get("PAPERPREP_TEXLIVE_BIN")
    if not texlive:
        pytest.skip("PAPERPREP_TEXLIVE_BIN not set — point at a TeX Live bin dir to run this suite")
    rc = run_paperprep_local(latex_manifest_module, out, texlive_bin=texlive)
    assert rc == 0, f"paperprep exited non-zero ({rc})"
    return out


# -----------------------------------------------------------------------------
# Compile-stage assertions (only meaningful for the _latex entries)
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("pid", [f"{a}_latex" for a in ARXIV_IDS])
def test_latex_compile_produced_pdf(latex_run_output, pid):
    pdf = latex_run_output / "compiled" / f"{pid}.pdf"
    assert pdf.exists(), f"compile stage failed to produce {pdf}"
    assert pdf.stat().st_size > 0


@pytest.mark.parametrize("pid", [f"{a}_latex" for a in ARXIV_IDS])
def test_latex_compile_page_count_matches_production(latex_run_output, pid):
    import fitz
    pdf = latex_run_output / "compiled" / f"{pid}.pdf"
    with fitz.open(str(pdf)) as doc:
        pp_pages = doc.page_count
    aid = pid.removesuffix("_latex").removeprefix("arxiv_")
    prod_count = int((GROUND_TRUTH_DIR / f"arxiv_{aid}" / "production_pdf_page_count.txt").read_text().strip())
    diff = abs(pp_pages - prod_count)
    assert diff <= PDF_PAGE_COUNT_TOLERANCE, (
        f"{pid}: compiled PDF has {pp_pages} pages, production has {prod_count} "
        f"(diff {diff} > tolerance {PDF_PAGE_COUNT_TOLERANCE})"
    )


@pytest.mark.parametrize("pid", [f"{a}_latex" for a in ARXIV_IDS])
def test_latex_apply_venue_toggle_ran(latex_run_output, pid):
    """`extra.toggle_changes > 0` proves apply_venue_toggle anonymized the .tex."""
    state = load_state(latex_run_output)
    compile_row = state_by(state, pid, "compile")
    assert compile_row is not None and compile_row["status"] == "ok"
    toggle_changes = (compile_row.get("extra") or {}).get("toggle_changes", 0)
    assert toggle_changes > 0, (
        f"{pid}: toggle_changes=0 — apply_venue_toggle didn't anonymize any \\author/\\institute blocks"
    )


@pytest.mark.parametrize("pid", [f"{a}_latex" for a in ARXIV_IDS])
def test_latex_scrub_tex_ran(latex_run_output, pid):
    """`extra.scrub_stats` proves scrub_tex_dir was invoked.

    Note: all-zero stats are valid — a clean paper with no URL leaks /
    ack sections / leaky footnotes correctly reports nothing-to-do. We
    only assert the function ran and reported the standard schema, not
    that it actually rewrote anything.
    """
    state = load_state(latex_run_output)
    compile_row = state_by(state, pid, "compile")
    scrub = (compile_row.get("extra") or {}).get("scrub_stats") or {}
    expected_keys = {
        "files_changed", "ack_sections_removed", "repro_sections_removed",
        "checklist_sections_removed", "leaky_footnotes_removed",
        "urls_stripped", "hrefs_unwrapped",
    }
    assert expected_keys.issubset(scrub.keys()), (
        f"{pid}: scrub_tex_dir didn't produce the expected stats dict; got {scrub}"
    )
    # All values should be ints (proves the schema; doesn't require any rewrite)
    for k in expected_keys:
        assert isinstance(scrub[k], int), f"{pid}: scrub_stats[{k}] is not an int: {scrub[k]!r}"


# -----------------------------------------------------------------------------
# Downstream stages on the paperprep-compiled PDFs work end-to-end
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("pid", [f"{a}_latex" for a in ARXIV_IDS])
def test_latex_full_pipeline_passes_for_every_paper(latex_run_output, pid):
    state = load_state(latex_run_output)
    for stage in ("mineru", "normalize", "filter", "export"):
        row = state_by(state, pid, stage)
        assert row is not None, f"{pid}: missing {stage} state row"
        assert row["status"] == "ok", f"{pid} {stage} failed: {row.get('error')}"
    # And the redacted PDF + normalized MD must be on disk
    assert (latex_run_output / "normalized" / pid / f"{pid}.md").exists()
    assert (latex_run_output / "normalized" / pid / f"{pid}_redacted.pdf").exists()


@pytest.mark.parametrize("pid", [f"{a}_latex" for a in ARXIV_IDS])
def test_latex_sharegpt_entry_present(latex_run_output, pid):
    text_data = json.loads((latex_run_output / "sharegpt" / "text" / "data.json").read_text())
    vision_data = json.loads((latex_run_output / "sharegpt" / "vision" / "data.json").read_text())
    text_ids = {e["_metadata"]["id"] for e in text_data}
    vision_ids = {e["_metadata"]["id"] for e in vision_data}
    assert pid in text_ids, f"{pid} missing from text/data.json"
    assert pid in vision_ids, f"{pid} missing from vision/data.json"


# -----------------------------------------------------------------------------
# All 9 entries (3 iclr pdf + 3 arxiv pdf + 3 arxiv latex) survive end-to-end
# -----------------------------------------------------------------------------

def test_all_nine_entries_in_sharegpt(latex_run_output):
    expected_ids = set(ICLR_IDS) | {f"{a}_pdf" for a in ARXIV_IDS} | {f"{a}_latex" for a in ARXIV_IDS}
    text_data = json.loads((latex_run_output / "sharegpt" / "text" / "data.json").read_text())
    text_ids = {e["_metadata"]["id"] for e in text_data}
    assert text_ids == expected_ids, (
        f"missing: {expected_ids - text_ids}; unexpected: {text_ids - expected_ids}"
    )
