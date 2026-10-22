"""Tests for paperprep.state — append-only StateLog."""
import json
import time

import pytest

from paper_anonymizer.paperprep.state import StageResult, StateLog


def test_append_then_read_back(tmp_path):
    log = StateLog(tmp_path / "state.jsonl")
    log.append(StageResult(id="x", stage="compile", status="ok",
                           output_path="/p/x.pdf"))
    log.append(StageResult(id="y", stage="compile", status="failed", error="oops"))
    rows = [json.loads(l) for l in (tmp_path / "state.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    assert rows[0]["status"] == "ok"
    assert rows[1]["error"] == "oops"


def test_already_ok_after_append(tmp_path):
    log = StateLog(tmp_path / "s.jsonl")
    assert not log.already_ok("x", "compile")
    log.append(StageResult(id="x", stage="compile", status="ok"))
    assert log.already_ok("x", "compile")
    assert not log.already_ok("y", "compile")
    assert not log.already_ok("x", "mineru")


def test_filter_unprocessed(tmp_path):
    log = StateLog(tmp_path / "s.jsonl")
    log.append(StageResult(id="a", stage="compile", status="ok"))
    log.append(StageResult(id="b", stage="compile", status="failed"))
    out = log.filter_unprocessed(["a", "b", "c"], "compile")
    assert out == ["b", "c"]


def test_summarize(tmp_path):
    log = StateLog(tmp_path / "s.jsonl")
    log.append(StageResult(id="a", stage="compile", status="ok"))
    log.append(StageResult(id="b", stage="compile", status="ok"))
    log.append(StageResult(id="c", stage="compile", status="failed"))
    log.append(StageResult(id="a", stage="mineru", status="ok"))
    s = log.summarize()
    assert s["compile"] == {"ok": 2, "failed": 1}
    assert s["mineru"] == {"ok": 1}


def test_log_recovers_from_existing_file(tmp_path):
    p = tmp_path / "s.jsonl"
    log1 = StateLog(p)
    log1.append(StageResult(id="a", stage="compile", status="ok"))
    # Re-open
    log2 = StateLog(p)
    assert log2.already_ok("a", "compile")
