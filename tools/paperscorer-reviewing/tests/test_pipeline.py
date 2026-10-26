"""Pipeline / server unit tests. No GPU, no paperprep/paperlens daemons --
we mock the upstream HTTP calls so the wiring stays under test.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

from paperlensreview.pipeline import (
    STAGES, JobRegistry, JobStatus, _decision_from_p_accept,
    _load_sharegpt_export, run_job,
)


# ---------------------------------------------------------------------------
# Pure-function tests
# ---------------------------------------------------------------------------

def test_decision_threshold():
    assert _decision_from_p_accept(0.51) == "Accept"
    assert _decision_from_p_accept(0.49) == "Reject"
    assert _decision_from_p_accept(0.5)  == "Accept"  # tie goes to accept


def test_sharegpt_export_appends_gpt_turn(tmp_path: Path):
    """paperprep emits [system, human] only. _load_sharegpt_export must
    append a placeholder gpt turn so LF's 'ppo' stage doesn't drop the row."""
    sg_path = tmp_path / "data.json"
    sg_path.write_text(json.dumps([{
        "conversations": [
            {"from": "system", "value": "sys"},
            {"from": "human",  "value": "user"},
        ],
        "_metadata": {"id": "x"},
        "images":    ["/x/page_1.png"],
    }]))
    body = {"sharegpt_vision_path": str(sg_path)}

    out = _load_sharegpt_export(body, "vision")
    assert out is not None
    roles = [c["from"] for c in out["conversations"]]
    assert roles == ["system", "human", "gpt"]

    # Don't overwrite an existing gpt turn
    rows = json.loads(sg_path.read_text())
    rows[0]["conversations"].append({"from": "gpt", "value": "real"})
    sg_path.write_text(json.dumps(rows))
    out2 = _load_sharegpt_export(body, "vision")
    assert out2["conversations"][-1] == {"from": "gpt", "value": "real"}


def test_sharegpt_export_missing_path():
    """Missing key -> None; the caller surfaces the error."""
    assert _load_sharegpt_export({}, "vision") is None
    assert _load_sharegpt_export({"sharegpt_vision_path": "/nonexistent/x"}, "vision") is None


def test_registry_create_get():
    r = JobRegistry()
    s = r.create("abc")
    assert isinstance(s, JobStatus)
    assert s.job_id == "abc"
    assert r.get("abc") is s
    assert r.get("missing") is None
    assert len(r.all()) == 1


# ---------------------------------------------------------------------------
# Integration: run_job with mocked paperprep-serve + paperlens-serve
# ---------------------------------------------------------------------------

def _make_cfg() -> OmegaConf:
    return OmegaConf.create({
        "server":          {"host": "127.0.0.1", "port": 0},
        "paperlens_serve": {"base_url": "http://paperlens.local", "timeout_seconds": 5},
        "paperprep_serve": {"base_url": "http://paperprep.local", "timeout_seconds": 5,
                            "work_dir": ""},
        "review":          {"modality": "vision", "domain": "arxiv"},
    })


def _ok_prepare_body(sharegpt_path: Path) -> dict:
    return {
        "request_id": "j",
        "output_dir": str(sharegpt_path.parent),
        "sharegpt_vision_path": str(sharegpt_path),
        "sharegpt_text_path": None,
        "papers": [{"id": "j", "status": "ok", "body_pages": 10}],
        "elapsed_s": 0.42,
    }


def _post_responder(prepare_body: dict, score_body: dict):
    def fake_post(url, json=None, timeout=None):
        class _R:
            status_code = 200
            def raise_for_status(self): pass
        if url.endswith("/prepare"):
            r = _R(); r.json = lambda: prepare_body
            return r
        if url.endswith("/score"):
            r = _R(); r.json = lambda: score_body
            return r
        raise AssertionError(f"unexpected POST {url}")
    return fake_post


def test_run_job_happy_path(tmp_path: Path):
    cfg = _make_cfg()
    work = tmp_path / "work"; work.mkdir()
    sg = tmp_path / "sharegpt_vision.json"
    sg.write_text(json.dumps([{
        "conversations": [
            {"from": "system", "value": "sys"},
            {"from": "human",  "value": "user"},
        ],
        "_metadata": {"id": "j"},
        "images":    [],
    }]))

    prepare_body = _ok_prepare_body(sg)
    score_body = {"scores": [{
        "p_accept": 0.77, "logp_accept": -0.3, "logp_reject": -1.5,
        "pred": "Outcome: \\boxed{Accept}",
    }]}

    status = JobStatus(job_id="j")
    with patch("paperlensreview.pipeline.requests.post",
               side_effect=_post_responder(prepare_body, score_body)):
        run_job(cfg, status, work, b"%PDF-1.4 fake", "fake.pdf")

    assert status.state == "done", status.error
    assert status.stage == "done"
    assert status.stage_index == len(STAGES)
    assert status.result is not None
    assert status.result["decision"] == "Accept"
    assert abs(status.result["p_accept"] - 0.77) < 1e-9
    assert status.result["paperprep_elapsed_s"] == 0.42


def test_run_job_paperprep_paper_failure(tmp_path: Path):
    """paperprep serve returned 200 but the per-paper status is failed."""
    cfg = _make_cfg()
    work = tmp_path / "work"; work.mkdir()

    prepare_body = {
        "request_id": "boom",
        "output_dir": str(tmp_path),
        "papers": [{"id": "boom", "status": "failed", "error": "mineru: blew up"}],
        "elapsed_s": 0.1,
    }
    status = JobStatus(job_id="boom")
    with patch("paperlensreview.pipeline.requests.post",
               side_effect=_post_responder(prepare_body, {})):
        run_job(cfg, status, work, b"%PDF-1.4 fake", "fake.pdf")
    assert status.state == "error"
    assert "mineru: blew up" in (status.error or "")


def test_run_job_export_missing(tmp_path: Path):
    """paperprep serve returned ok but no sharegpt path -> mark error."""
    cfg = _make_cfg()
    work = tmp_path / "work"; work.mkdir()

    prepare_body = {
        "request_id": "j",
        "output_dir": str(tmp_path),
        "papers": [{"id": "j", "status": "ok", "body_pages": 10}],
        "elapsed_s": 0.1,
        # NO sharegpt_vision_path
    }
    status = JobStatus(job_id="j")
    with patch("paperlensreview.pipeline.requests.post",
               side_effect=_post_responder(prepare_body, {})):
        run_job(cfg, status, work, b"%PDF-1.4 fake", "fake.pdf")
    assert status.state == "error"
    assert "sharegpt row" in (status.error or "")
