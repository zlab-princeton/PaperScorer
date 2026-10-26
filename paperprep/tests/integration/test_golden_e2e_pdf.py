"""Ground-truth correctness test — PDF inputs only.

Runs paperprep end-to-end on 6 PDFs (3 ICLR + 3 arxiv) and verifies the
outputs match production-pipeline ground-truth stored under
``tests/fixtures/golden/ground_truth/``.

Tagged ``gpu`` because the MinerU stage needs a CUDA-capable GPU. Run with:

    uv run pytest tests/integration/test_golden_e2e_pdf.py -m gpu --no-header -s

Wall time: ~20 min on an L40 (MinerU is the long pole).
"""
from __future__ import annotations

import difflib
import json
from collections import Counter
from pathlib import Path

import pytest

from tests.integration.conftest import (
    ALL_PDF_IDS,
    ARXIV_IDS,
    ICLR_IDS,
    GROUND_TRUTH_DIR,
    load_ground_truth_json,
    load_ground_truth_text,
    load_state,
    run_paperprep_local,
    state_by,
)


# Similarity thresholds calibrated from the human-validated diff runs.
# arxiv = production used the same prompt + filter mode, so very close to identical.
# ICLR = production L2 includes the appendix; paperprep cuts at refs (matching arxiv
# pipeline) — so similarity is lower by design but structure matches exactly.
ARXIV_MD_SIM_MIN = 0.90
ICLR_MD_SIM_MIN = 0.60          # design-difference appendix handling
ARXIV_TEXT_BODY_SIM_MIN = 0.90
ARXIV_MINERU_BLOCK_TOLERANCE = 5
ARXIV_MINERU_TEXT_OVERLAP_MIN = 0.85


pytestmark = [pytest.mark.gpu]


@pytest.fixture(scope="module")
def pdf_run_output(tmp_path_factory, mineru_available, pdf_manifest_module):
    """Run paperprep on the 6-PDF manifest exactly once, share output across tests."""
    out = tmp_path_factory.mktemp("pdf_run_out")
    rc = run_paperprep_local(pdf_manifest_module, out)
    assert rc == 0, f"paperprep exited non-zero ({rc})"
    return out


@pytest.fixture(scope="module")
def pdf_manifest_module(tmp_path_factory):
    """Module-scoped variant of the pdf_manifest fixture (needs to survive across tests)."""
    import json as _j
    from tests.integration.conftest import INPUTS_DIR
    tp = tmp_path_factory.mktemp("manifest_pdf")
    p = tp / "manifest.jsonl"
    with open(p, "w") as f:
        for pid in ALL_PDF_IDS:
            f.write(_j.dumps({"id": pid, "type": "pdf", "path": str(INPUTS_DIR / f"{pid}.pdf")}) + "\n")
    return p


# -----------------------------------------------------------------------------
# 1. State assertions — every paper through every stage
# -----------------------------------------------------------------------------

def test_state_all_stages_ok(pdf_run_output):
    state = load_state(pdf_run_output)
    for pid in ALL_PDF_IDS:
        for stage in ("mineru", "normalize", "filter", "export"):
            row = state_by(state, pid, stage)
            assert row is not None, f"no {stage} state row for {pid}"
            assert row["status"] == "ok", f"{pid} {stage} not ok: {row.get('error')}"
        # compile should be skipped for type=pdf
        compile_row = state_by(state, pid, "compile")
        assert compile_row is not None and compile_row["status"] == "skipped"


# -----------------------------------------------------------------------------
# 2. Redacted PDF exists for every paper (this is the regression gate for the
#    redact_pdf 'int not iterable' bug we fixed)
# -----------------------------------------------------------------------------

def test_redacted_pdf_produced_for_every_paper(pdf_run_output):
    for pid in ALL_PDF_IDS:
        pdf = pdf_run_output / "normalized" / pid / f"{pid}_redacted.pdf"
        assert pdf.exists(), f"redact_pdf failed to produce {pdf}"
        assert pdf.stat().st_size > 0, f"empty redacted pdf: {pdf}"


# -----------------------------------------------------------------------------
# 3. MinerU content_list comparison vs production (arxiv only — ICLR's
#    production MinerU output isn't co-located in the same dir structure)
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("pid", ARXIV_IDS)
def test_arxiv_mineru_matches_production(pdf_run_output, pid):
    paperprep_cl_path = next((pdf_run_output / "mineru" / pid).rglob("*_content_list.json"))
    paperprep_cl = json.loads(paperprep_cl_path.read_text())

    aid = pid.removeprefix("arxiv_")
    prod_cl = load_ground_truth_json(f"arxiv_{aid}", "mineru_content_list")

    # Block count: vLLM can re-chunk text blocks slightly across runs
    diff = abs(len(paperprep_cl) - len(prod_cl))
    assert diff <= ARXIV_MINERU_BLOCK_TOLERANCE, (
        f"{pid}: block count diff {diff} > tolerance {ARXIV_MINERU_BLOCK_TOLERANCE} "
        f"(paperprep={len(paperprep_cl)}, prod={len(prod_cl)})"
    )

    # Type-set must match (modulo content-dependent types like 'code')
    pp_types = set(b.get("type") for b in paperprep_cl)
    prod_types = set(b.get("type") for b in prod_cl)
    # Allow one-sided differences for content-dependent types (code, page_footnote)
    optional_types = {"code", "page_footnote", "header"}
    core_pp = pp_types - optional_types
    core_prod = prod_types - optional_types
    assert core_pp == core_prod, f"{pid}: core type set diverges: pp={core_pp}, prod={core_prod}"

    # Text-block overlap
    pp_text = set(b.get("text", "").strip() for b in paperprep_cl if b.get("type") == "text")
    prod_text = set(b.get("text", "").strip() for b in prod_cl if b.get("type") == "text")
    overlap = len(pp_text & prod_text)
    base = max(len(pp_text), len(prod_text))
    overlap_ratio = overlap / base if base else 0
    assert overlap_ratio >= ARXIV_MINERU_TEXT_OVERLAP_MIN, (
        f"{pid}: text overlap {overlap_ratio:.3f} < threshold {ARXIV_MINERU_TEXT_OVERLAP_MIN}"
    )


# -----------------------------------------------------------------------------
# 4. Normalized markdown similarity vs production
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("pid", ARXIV_IDS)
def test_arxiv_normalized_md_similarity(pdf_run_output, pid):
    paperprep_md = (pdf_run_output / "normalized" / pid / f"{pid}.md").read_text()
    aid = pid.removeprefix("arxiv_")
    prod_md = load_ground_truth_text(f"arxiv_{aid}", "normalized")
    ratio = difflib.SequenceMatcher(None, paperprep_md, prod_md).ratio()
    assert ratio >= ARXIV_MD_SIM_MIN, (
        f"{pid}: md similarity {ratio:.3f} < threshold {ARXIV_MD_SIM_MIN}"
    )


@pytest.mark.parametrize("pid", ICLR_IDS)
def test_iclr_normalized_md_similarity(pdf_run_output, pid):
    paperprep_md = (pdf_run_output / "normalized" / pid / f"{pid}.md").read_text()
    sid = pid.removeprefix("iclr_")
    prod_md = load_ground_truth_text(f"iclr_{sid}", "normalized_l2")
    ratio = difflib.SequenceMatcher(None, paperprep_md, prod_md).ratio()
    assert ratio >= ICLR_MD_SIM_MIN, (
        f"{pid}: md similarity {ratio:.3f} < threshold {ICLR_MD_SIM_MIN}"
    )
    # Even though structure differs (appendix), the title + abstract should be present
    assert "ABSTRACT" in paperprep_md or "Abstract" in paperprep_md
    assert "Anonymous authors" in paperprep_md or "anonymous" in paperprep_md.lower()


# -----------------------------------------------------------------------------
# 5. ShareGPT entry shape + text-body similarity vs production
# -----------------------------------------------------------------------------

def test_sharegpt_files_exist_and_have_all_papers(pdf_run_output):
    text_data = json.loads((pdf_run_output / "sharegpt" / "text" / "data.json").read_text())
    vision_data = json.loads((pdf_run_output / "sharegpt" / "vision" / "data.json").read_text())
    text_ids = {e["_metadata"]["id"] for e in text_data}
    vision_ids = {e["_metadata"]["id"] for e in vision_data}
    assert text_ids == set(ALL_PDF_IDS), f"text data missing ids: {set(ALL_PDF_IDS) - text_ids}"
    assert vision_ids == set(ALL_PDF_IDS), f"vision data missing ids: {set(ALL_PDF_IDS) - vision_ids}"


@pytest.mark.parametrize("pid", ALL_PDF_IDS)
def test_sharegpt_inference_only_roles(pdf_run_output, pid):
    """Inference-only mode: [system, human] roles, no gpt turn."""
    text_data = json.loads((pdf_run_output / "sharegpt" / "text" / "data.json").read_text())
    vision_data = json.loads((pdf_run_output / "sharegpt" / "vision" / "data.json").read_text())
    text_entry = next(e for e in text_data if e["_metadata"]["id"] == pid)
    vision_entry = next(e for e in vision_data if e["_metadata"]["id"] == pid)
    assert [c["from"] for c in text_entry["conversations"]] == ["system", "human"]
    assert [c["from"] for c in vision_entry["conversations"]] == ["system", "human"]


@pytest.mark.parametrize("pid", ARXIV_IDS)
def test_arxiv_sharegpt_text_body_similarity(pdf_run_output, pid):
    text_data = json.loads((pdf_run_output / "sharegpt" / "text" / "data.json").read_text())
    paperprep_entry = next(e for e in text_data if e["_metadata"]["id"] == pid)
    aid = pid.removeprefix("arxiv_")
    prod_entry = load_ground_truth_json(f"arxiv_{aid}", "sharegpt_text_entry")
    pp_body = paperprep_entry["conversations"][1]["value"]
    prod_body = prod_entry["conversations"][1]["value"]
    # Strip the prompt prefix (after "rates\n\n") to compare just the paper body
    sep = "rates\n\n"
    pp_paper = pp_body.split(sep, 1)[-1]
    prod_paper = prod_body.split(sep, 1)[-1]
    ratio = difflib.SequenceMatcher(None, pp_paper, prod_paper).ratio()
    assert ratio >= ARXIV_TEXT_BODY_SIM_MIN, (
        f"{pid}: sharegpt text-body similarity {ratio:.3f} < {ARXIV_TEXT_BODY_SIM_MIN}"
    )


# -----------------------------------------------------------------------------
# 6. Vision image-count alignment vs production
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("pid", ARXIV_IDS)
def test_arxiv_vision_image_count_matches_production(pdf_run_output, pid):
    vision_data = json.loads((pdf_run_output / "sharegpt" / "vision" / "data.json").read_text())
    paperprep_entry = next(e for e in vision_data if e["_metadata"]["id"] == pid)
    aid = pid.removeprefix("arxiv_")
    prod_entry = load_ground_truth_json(f"arxiv_{aid}", "sharegpt_vision_entry")
    pp_n = len(paperprep_entry.get("images", []))
    prod_n = len(prod_entry.get("images", []))
    assert pp_n == prod_n, f"{pid}: vision image count {pp_n} != production {prod_n}"
    # And the human turn's <image>-token count must match images list length
    human_text = paperprep_entry["conversations"][1]["value"]
    assert human_text.count("<image>") == pp_n
