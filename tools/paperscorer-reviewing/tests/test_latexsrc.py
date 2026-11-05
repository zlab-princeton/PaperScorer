"""Tests for the LaTeX-source input mode: git/main.tex helpers (latexsrc) and
the run_latex_job / run_latex_history_job pipeline paths.

No GPU, no paperprep/paperlens daemons -- the upstream HTTP calls are mocked.
``git archive`` runs for real against a throwaway repo built in a tmp dir, so the
commit -> tree -> prepare wiring is exercised end-to-end.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

from paperlensreview import latexsrc
from paperlensreview.pipeline import (
    JobStatus, run_latex_job, run_latex_history_job,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def _commit(repo: Path, fname: str, body: str, msg: str) -> None:
    (repo / fname).write_text(body)
    _git(repo, "add", fname)
    _git(repo, "-c", "user.email=t@t.com", "-c", "user.name=t", "commit", "-q", "-m", msg)


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """A repo with 3 commits of differing .tex churn (oldest->newest):
    c1 big, c2 tiny, c3 medium."""
    repo = tmp_path / "paper"
    repo.mkdir()
    _git(repo, "init", "-q")
    head = "\\documentclass{article}\n\\begin{document}\n"
    tail = "\\end{document}\n"
    _commit(repo, "main.tex", head + "x\n" * 40 + tail, "c1 big")
    _commit(repo, "main.tex", head + "x\n" * 41 + tail, "c2 tiny")
    _commit(repo, "main.tex", head + "y\n" * 60 + tail, "c3 med")
    return repo


def _make_cfg() -> OmegaConf:
    return OmegaConf.create({
        "paperlens_serve": {"base_url": "http://paperlens.local", "timeout_seconds": 5},
        "paperprep_serve": {"base_url": "http://paperprep.local", "timeout_seconds": 5, "work_dir": ""},
        "review":          {"modality": "vision", "domain": "arxiv"},
    })


def _responder(prepare_body: dict, score_body: dict):
    def fake_post(url, json=None, timeout=None):
        class _R:
            status_code = 200
            def raise_for_status(self): pass
        r = _R()
        if url.endswith("/prepare"):
            r.json = lambda: prepare_body
        elif url.endswith("/score"):
            r.json = lambda: score_body
        else:
            raise AssertionError(f"unexpected POST {url}")
        return r
    return fake_post


# ---------------------------------------------------------------------------
# latexsrc unit tests
# ---------------------------------------------------------------------------

def test_percentile_interpolation():
    assert latexsrc._percentile([], 25) == 0.0
    assert latexsrc._percentile([5], 25) == 5.0
    # sorted [1,43,101], rank=0.5 -> 1 + 0.5*(43-1) = 22
    assert latexsrc._percentile([43, 1, 101], 25) == pytest.approx(22.0)


def test_find_main_tex(git_repo: Path):
    assert latexsrc.find_main_tex(git_repo) == "main.tex"
    assert latexsrc.list_tex_files(git_repo) == ["main.tex"]


def test_is_git_repo_and_resolve(git_repo: Path, tmp_path: Path):
    assert latexsrc.is_git_repo(git_repo) is True
    assert latexsrc.is_git_repo(tmp_path / "nope") is False
    assert latexsrc.resolve_commit(git_repo, "HEAD") is not None
    assert latexsrc.resolve_commit(git_repo, "deadbeef") is None


def test_churn_ranking_and_pct(git_repo: Path):
    # Default cutoff is p50 (median). For 3 commits with churns [43, 1, 101],
    # sorted = [1, 43, 101], median = 43. churn >= 43 picks the big and the
    # medium; the tiny (churn ~1) falls below.
    h = latexsrc.last_commits_with_tex_churn(git_repo, n=20)
    assert h["n"] == 3
    assert h["pct"] == 50.0
    # oldest -> newest order preserved
    assert [c["subject"] for c in h["commits"]] == ["c1 big", "c2 tiny", "c3 med"]
    by_subj = {c["subject"]: c for c in h["commits"]}
    assert by_subj["c2 tiny"]["churn"] < by_subj["c1 big"]["churn"]
    assert by_subj["c2 tiny"]["churn"] < by_subj["c3 med"]["churn"]
    assert by_subj["c2 tiny"]["above_pct"] is False
    assert by_subj["c1 big"]["above_pct"] is True
    assert by_subj["c3 med"]["above_pct"] is True
    assert all(len(c["short"]) == 8 for c in h["commits"])
    assert h["pct_churn"] >= 0
    # Tune the percentile -- at p100, only the single max-churn commit qualifies.
    h2 = latexsrc.last_commits_with_tex_churn(git_repo, n=20, pct=100.0)
    assert sum(1 for c in h2["commits"] if c["above_pct"]) == 1
    # At p0, all commits qualify (churn >= min).
    h3 = latexsrc.last_commits_with_tex_churn(git_repo, n=20, pct=0.0)
    assert all(c["above_pct"] for c in h3["commits"])


def test_archive_commit_isolated(git_repo: Path, tmp_path: Path):
    sha = latexsrc.resolve_commit(git_repo, "HEAD")
    dest = tmp_path / "extracted"
    latexsrc.archive_commit(git_repo, sha, dest)
    assert (dest / "main.tex").is_file()
    assert "y" in (dest / "main.tex").read_text()  # HEAD = c3 (the y-lines)


# ---------------------------------------------------------------------------
# pipeline: run_latex_job (working tree, single verdict)
# ---------------------------------------------------------------------------

def test_run_latex_job_happy(git_repo: Path, tmp_path: Path):
    cfg = _make_cfg()
    work = tmp_path / "work"; work.mkdir()
    sg = tmp_path / "sg.json"
    sg.write_text(json.dumps([{
        "conversations": [{"from": "system", "value": "s"}, {"from": "human", "value": "u"}],
        "_metadata": {"id": "j"}, "images": [],
    }]))
    prepare_body = {
        "output_dir": str(tmp_path / "ppout"),
        "sharegpt_vision_path": str(sg),
        "papers": [{"id": "j", "status": "ok", "body_pages": 8}],
        "elapsed_s": 0.3,
    }
    score_body = {"scores": [{"p_accept": 0.81, "logp_accept": -0.2, "logp_reject": -1.7}]}

    status = JobStatus(job_id="j")
    with patch("paperlensreview.pipeline.requests.post",
               side_effect=_responder(prepare_body, score_body)):
        run_latex_job(cfg, status, work, git_repo, "main.tex")

    assert status.state == "done", status.error
    assert status.result["source_type"] == "latex_dir"
    assert status.result["git_mode"] == "latest"
    assert status.result["main_tex"] == "main.tex"
    assert status.result["decision"] == "Accept"
    assert status.result["p_accept"] == pytest.approx(0.81)


# ---------------------------------------------------------------------------
# pipeline: run_latex_history_job (multi-commit trajectory, real git archive)
# ---------------------------------------------------------------------------

def test_run_latex_history_job(git_repo: Path, tmp_path: Path):
    cfg = _make_cfg()
    work = tmp_path / "work"; work.mkdir()
    sg = tmp_path / "sg.json"   # persistent; NOT under the (deletable) output_dir
    sg.write_text(json.dumps([{
        "conversations": [{"from": "system", "value": "s"}, {"from": "human", "value": "u"}],
        "_metadata": {"id": "j"}, "images": [],
    }]))
    prepare_body = {
        "output_dir": str(tmp_path / "ppout"),   # safe to rmtree between commits
        "sharegpt_vision_path": str(sg),
        "papers": [{"id": "j", "status": "ok", "body_pages": 8}],
        "elapsed_s": 0.1,
    }
    score_body = {"scores": [{"p_accept": 0.6, "logp_accept": -0.5, "logp_reject": -0.9}]}

    h = latexsrc.last_commits_with_tex_churn(git_repo, n=20)
    selected = [c for c in h["commits"] if c["above_pct"]]   # c1 big, c3 med
    assert len(selected) == 2

    status = JobStatus(job_id="hist")
    with patch("paperlensreview.pipeline.requests.post",
               side_effect=_responder(prepare_body, score_body)):
        run_latex_history_job(cfg, status, work, git_repo, "main.tex", selected)

    assert status.state == "done", status.error
    res = status.result
    assert res["source_type"] == "latex_history"
    assert res["n_commits"] == 2
    assert res["n_scored"] == 2
    assert len(res["trajectory"]) == 2
    assert all(r["state"] == "done" for r in res["trajectory"])
    assert all(r["p_accept"] == pytest.approx(0.6) for r in res["trajectory"])
    # trajectory keeps oldest->newest order from the selection
    assert [r["subject"] for r in res["trajectory"]] == ["c1 big", "c3 med"]
    # Per-commit artifacts are KEPT (browseable via the drilldown panel):
    # both the archived src tree and the paperprep output dir survive, and
    # the trajectory rec records their paths.
    for rec in res["trajectory"]:
        assert (work / "hist" / "src" / rec["short"]).exists(), "src tree should persist"
        assert rec["src_dir"] == str(work / "hist" / "src" / rec["short"])
        assert rec["paperprep_output_dir"]  # not popped after scoring


def test_list_dirs_endpoint(git_repo: Path, tmp_path: Path):
    """Server-side folder browser flags git/tex dirs and hides dotdirs."""
    from fastapi.testclient import TestClient
    import paperlensreview.server as s

    (tmp_path / "plain").mkdir()
    (tmp_path / ".hidden").mkdir()
    with TestClient(s.app) as c:
        j = c.post("/list_dirs", json={"path": str(tmp_path)}).json()
    names = {d["name"]: d for d in j["dirs"]}
    assert "paper" in names                      # the git_repo fixture dir
    assert names["paper"]["is_git"] and names["paper"]["has_tex"]
    assert "plain" in names and not names["plain"]["is_git"]
    assert ".hidden" not in names                # dotdirs hidden
    assert j["parent"] is not None
    assert j["path"] == str(tmp_path.resolve())


def test_run_latex_history_records_failures(git_repo: Path, tmp_path: Path):
    """A per-commit paperprep failure is recorded, not fatal to the job."""
    cfg = _make_cfg()
    work = tmp_path / "work"; work.mkdir()
    prepare_body = {
        "output_dir": str(tmp_path / "ppout"),
        "papers": [{"id": "j", "status": "failed", "error": "compile: boom"}],
        "elapsed_s": 0.1,
    }
    h = latexsrc.last_commits_with_tex_churn(git_repo, n=20)
    selected = [c for c in h["commits"] if c["above_pct"]]

    status = JobStatus(job_id="histfail")
    with patch("paperlensreview.pipeline.requests.post",
               side_effect=_responder(prepare_body, {})):
        run_latex_history_job(cfg, status, work, git_repo, "main.tex", selected)

    assert status.state == "done"                       # job completes
    assert status.result["n_scored"] == 0               # but nothing scored
    assert all(r["state"] == "error" for r in status.result["trajectory"])
    assert "compile: boom" in status.result["trajectory"][0]["error"]
