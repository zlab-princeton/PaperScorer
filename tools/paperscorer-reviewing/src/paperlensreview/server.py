"""FastAPI server for paperlensreview.

Endpoints:
  POST /submit              upload a PDF -> {job_id}
  GET  /status/{job_id}     poll for progress + result
  GET  /health              service + upstream healthcheck
  GET  /                    minimal HTML UI

The pipeline runs in a background thread (one job at a time per worker --
MinerU + paperlens both want GPU). For higher throughput, run multiple
paperlensreview workers behind a queue.
"""
from __future__ import annotations

import logging
import os
import threading
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from omegaconf import OmegaConf
from pydantic import BaseModel

from .checks import check_url_health
from . import arxiv as _arxiv
from . import latexsrc
from . import agents as _agents_mod
from .pipeline import (
    JobRegistry, JobStatus, STAGES, STAGES_AGENT,
    run_job, run_latex_job, run_latex_history_job,
    _enter_stage,
)


# /submit_arxiv prepends two stages to the standard pipeline so the UI's
# stage bar shows fetch + extract before paperprep+score.
STAGES_ARXIV = ["download", "latex_extraction"] + STAGES


def _stages_with_agent(base: list[str], agent: Optional[str]) -> list[str]:
    """Append the two agent stages when the user opted in. Mirrors what
    _run_agent_stage emits in the pipeline so the UI's stage bar lines up."""
    return list(base) + (list(STAGES_AGENT) if agent else [])


def _normalize_agent(choice: Optional[str]) -> Optional[str]:
    """Validate the user's agent toggle against probe results. Returns the
    canonical name (``"claude"`` / ``"codex"``) or None if the toggle was off
    or the requested agent isn't installed (we soft-ignore rather than 400 --
    the UI's checkbox already gated on the probe; an inconsistency here means
    the toggle raced the install state)."""
    if not choice:
        return None
    choice = choice.lower().strip()
    if choice not in ("claude", "codex"):
        return None
    probe = _agents_mod.probe_agents()
    if not probe.get(choice, {}).get("available"):
        log.warning("agent %r requested but probe says unavailable: %r", choice, probe.get(choice))
        return None
    return choice


log = logging.getLogger(__name__)


# --------- App + state ---------

_state: dict[str, Any] = {}
app = FastAPI(title="paperlens-reviewing")

_UI_DIR = Path(__file__).resolve().parent / "ui"


@app.on_event("startup")
def _startup() -> None:
    cfg_path = os.environ.get("PAPERLENSREVIEW_CONFIG", "configs/server.yaml")
    cfg = OmegaConf.load(cfg_path)
    if (env := os.environ.get("PAPERLENS_SERVE_BASE_URL")):
        cfg.paperlens_serve.base_url = env
    if (env := os.environ.get("PAPERPREP_SERVE_BASE_URL")):
        cfg.paperprep_serve.base_url = env
    # Modality/domain must match the loaded paperlens-serve model: a text model
    # (template=qwen) rejects the vision sharegpt row's <image> tags, so the
    # orchestrator sets these from the active model when it launches us.
    if (env := os.environ.get("PAPERLENS_REVIEW_MODALITY")):
        cfg.review.modality = env
    if (env := os.environ.get("PAPERLENS_REVIEW_DOMAIN")):
        cfg.review.domain = env
    log.info("[paperlensreview] loaded config from %s (modality=%s, domain=%s)",
             cfg_path, cfg.review.modality, cfg.review.domain)
    work_dir = Path(str(cfg.paperprep_serve.work_dir)).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    _state["cfg"] = cfg
    _state["work_dir"] = work_dir
    _state["jobs"] = JobRegistry()
    # Serialize submissions: paperprep serve + paperlens serve each hold a vLLM
    # engine on the same GPU; interleaved /prepare + /score calls thrash.
    _state["job_lock"] = threading.Lock()
    log.info("[paperlensreview] work_dir=%s paperlens=%s paperprep=%s",
             work_dir, cfg.paperlens_serve.base_url, cfg.paperprep_serve.base_url)


# --------- Endpoints ---------

@app.get("/")
def root() -> FileResponse:
    idx = _UI_DIR / "index.html"
    if not idx.exists():
        raise HTTPException(500, f"UI not bundled: {idx}")
    return FileResponse(idx)


@app.get("/health")
def health() -> dict:
    cfg = _state["cfg"]
    pl = check_url_health(cfg.paperlens_serve.base_url, path="/health")
    pp = check_url_health(cfg.paperprep_serve.base_url, path="/healthz")
    return {
        "status": "ok" if (pl.ok and pp.ok) else "degraded",
        "paperlens_serve": {"url": cfg.paperlens_serve.base_url, "healthy": pl.ok, "detail": pl.detail},
        "paperprep_serve": {"url": cfg.paperprep_serve.base_url, "healthy": pp.ok, "detail": pp.detail},
        "n_jobs": len(_state["jobs"].all()),
        "stages": STAGES,
        # Agent availability is probed lazily (cached process-lifetime). The UI
        # uses this to enable/disable the claude+codex radios on PDF + arxiv
        # tabs; the agent stages are entirely opt-in.
        "agents": _agents_mod.probe_agents(),
    }


@app.post("/submit")
def submit(file: UploadFile = File(...),
           agent: Optional[str] = Form(None),
           max_pages: Optional[int] = Form(None)) -> dict:
    """Accept a PDF upload, kick off the pipeline in a background thread,
    return the job_id immediately. The UI then polls /status/{job_id}.

    ``agent`` ("claude" | "codex" | None) opts into the post-decision agentic
    review. The two extra stages are appended to STAGES on the response so the
    UI's progress bar shows them from the start. Form(...) is required here
    because the UI posts multipart/form-data; without it FastAPI would treat
    ``agent`` as a query string param and silently drop the body field.
    ``max_pages`` overrides paperprep's page cap for this job (None = server
    default, currently 14).
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "only .pdf uploads are supported")
    pdf_bytes = file.file.read()
    if not pdf_bytes:
        raise HTTPException(400, "empty file")
    if len(pdf_bytes) > 100 * 1024 * 1024:
        raise HTTPException(413, f"PDF too large: {len(pdf_bytes)} bytes (>100 MB cap)")

    job_id = uuid.uuid4().hex[:12]
    status = _state["jobs"].create(job_id)
    agent_choice = _normalize_agent(agent)
    status.agent_choice = agent_choice
    stages = _stages_with_agent(STAGES, agent_choice)
    status.stages = list(stages)
    status.total_stages = len(stages)
    log.info("[/submit] job=%s file=%r size=%d agent=%s",
             job_id, file.filename, len(pdf_bytes), agent_choice)

    def _runner():
        # One pipeline at a time per worker process (MinerU + paperlens are
        # GPU-bound; they thrash if interleaved).
        with _state["job_lock"]:
            run_job(_state["cfg"], status, _state["work_dir"], pdf_bytes, file.filename,
                    max_pages=max_pages)

    t = threading.Thread(target=_runner, daemon=True, name=f"job-{job_id}")
    t.start()
    return {"job_id": job_id, "stages": stages, "submitted_at": status.started_at,
            "agent": agent_choice}


# --------- LaTeX-source input (local dir, optional git history) ---------

class ProbeLatexReq(BaseModel):
    path: str


class SubmitLatexReq(BaseModel):
    path: str
    main_tex: Optional[str] = None
    mode: str = "latest"               # "latest" (working tree) | "history"
    commits: Optional[list[dict]] = None  # selected commits for history mode
    n_window: int = 50                 # history default window (last N .tex commits)
    max_pages: Optional[int] = None    # override paperprep page cap (None = server default 14)


class ListDirsReq(BaseModel):
    path: Optional[str] = None


class SubmitArxivReq(BaseModel):
    arxiv_id: str                        # bare id (2305.00838) or any arxiv.org URL
    main_tex: Optional[str] = None       # optional entrypoint hint; auto-detected if omitted
    agent: Optional[str] = None          # "claude" | "codex" | None (off)
    max_pages: Optional[int] = None      # override paperprep page cap (None = server default 14)


@app.post("/submit_arxiv")
def submit_arxiv(req: SubmitArxivReq) -> dict:
    """Download an arxiv paper's LaTeX source, extract it, and route through the
    normal latex_dir review path. The server (paperlensreview) does the fetch
    so it needs internet — in slurm mode that means running on the login node,
    not inside the GPU allocation.
    """
    import time as _time
    try:
        aid = _arxiv.normalize_id(req.arxiv_id)
    except _arxiv.ArxivError as e:
        raise HTTPException(400, str(e))

    job_id = uuid.uuid4().hex[:12]
    status = _state["jobs"].create(job_id)
    agent_choice = _normalize_agent(req.agent)
    status.agent_choice = agent_choice
    log.info("[/submit_arxiv] job=%s arxiv_id=%s main_tex=%s agent=%s",
             job_id, aid, req.main_tex, agent_choice)

    # Pre-set the stage list so the UI's stage bar shows 4 boxes
    # (download -> latex_extraction -> paperprep -> paperlens_score) instead of 2.
    # _enter_stage uses status.stages to compute stage_index, so the indices
    # land correctly even though run_latex_job emits the trailing 2 stages.
    # When agent=claude|codex, the 2 trailing agent stages are appended too.
    arxiv_stages = _stages_with_agent(STAGES_ARXIV, agent_choice)
    status.stages = list(arxiv_stages)
    status.total_stages = len(arxiv_stages)

    def _runner():
        with _state["job_lock"]:
            job_dir = _state["work_dir"] / job_id
            job_dir.mkdir(parents=True, exist_ok=True)
            src_dir = job_dir / "arxiv_src"
            gz_path = job_dir / f"{aid.replace('/', '_')}.gz"
            status.state = "running"
            status.started_at = _time.time()
            try:
                _enter_stage(status, "download", via="arxiv")
                _arxiv.download_source(aid, gz_path)
                _enter_stage(status, "latex_extraction", via="arxiv")
                if not _arxiv.extract_source(gz_path, src_dir):
                    raise RuntimeError(f"could not extract {gz_path} (unrecognized archive shape)")
                main_tex = req.main_tex or latexsrc.find_main_tex(src_dir)
                # Hand off to the existing latex-dir pipeline. It calls
                # _enter_stage("paperprep") then ("paperlens_score") which
                # resolve to indices 2 / 3 against our 4-stage list.
                run_latex_job(_state["cfg"], status, _state["work_dir"], src_dir, main_tex,
                              max_pages=req.max_pages)
                # Mark the source on the result so the UI knows it was arxiv-sourced.
                if status.result is not None:
                    status.result["source_type"] = "latex_arxiv"
                    status.result["arxiv_id"] = aid
            except Exception as e:
                log.exception("[/submit_arxiv] job=%s failed", job_id)
                status.state = "error"
                status.error = f"arxiv fetch: {e}"
                status.finished_at = _time.time()

    t = threading.Thread(target=_runner, daemon=True, name=f"job-{job_id}")
    t.start()
    return {"job_id": job_id, "stages": arxiv_stages, "submitted_at": status.started_at,
            "arxiv_id": aid, "mode": "arxiv", "agent": agent_choice}


def _default_browse_path() -> Path:
    """Best initial folder for the dir browser. Mirrors the setup wizard's
    ``form._default_parent``: writable per-user scratch -> /scratch/gpfs -> $HOME.
    """
    import getpass
    import glob
    user = os.environ.get("USER") or getpass.getuser()
    for p in sorted(glob.glob(f"/scratch/gpfs/*/{user}")):
        if Path(p).is_dir():
            return Path(p)
    if Path("/scratch/gpfs").is_dir():
        return Path("/scratch/gpfs")
    return Path.home()


def _browse_bookmarks() -> list[dict]:
    """Quick-jump shortcuts shown above the dir list. Only entries that exist
    on this host are surfaced -- the dialog gracefully degrades on machines
    without /scratch/gpfs (laptops, CI runners)."""
    out = []
    sgu = _default_browse_path()
    if sgu != Path.home() and sgu != Path("/scratch/gpfs"):
        out.append({"label": "scratch (you)", "path": str(sgu)})
    if Path("/scratch/gpfs").is_dir():
        out.append({"label": "/scratch/gpfs", "path": "/scratch/gpfs"})
    out.append({"label": "home", "path": str(Path.home())})
    out.append({"label": "/", "path": "/"})
    return out


# --------- Try-me example assets (MAE, arXiv 2111.06377) ---------

# Bundled try-me assets live at the repo root under example_assets/mae/.
# Resolution: server.py is at <repo>/src/paperlensreview/server.py so
# parents[2] is the repo root (parents[0]=paperlensreview, [1]=src, [2]=repo).
# Works in both editable installs and direct-from-checkout runs.
# arXiv mode never needs a local asset -- it fetches live by id.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_EX_DIR = _REPO_ROOT / "example_assets" / "mae"
_EX_MAE_PDF = _EX_DIR / "mae_anonymized.pdf"
_EX_MAE_LATEX_DIR = _EX_DIR / "latex"
_EX_MAE_ARXIV_ID = "2111.06377"


@app.get("/examples")
def examples_index() -> dict:
    """Tell the UI which try-me Run buttons should be enabled. Each entry is
    {available: bool, label: str} so the UI can grey out missing assets on
    hosts where the bundled paths don't exist."""
    return {
        "mae_pdf": {
            "available": _EX_MAE_PDF.is_file(),
            "label": "MAE (anonymized PDF)",
            "size_bytes": _EX_MAE_PDF.stat().st_size if _EX_MAE_PDF.is_file() else None,
        },
        "mae_latex": {
            "available": _EX_MAE_LATEX_DIR.is_dir() and any(_EX_MAE_LATEX_DIR.glob("*.tex")),
            "label": "MAE (LaTeX source)",
            "path": str(_EX_MAE_LATEX_DIR) if _EX_MAE_LATEX_DIR.is_dir() else None,
        },
        "mae_arxiv": {
            "available": True,    # live download; the only failure mode is the network
            "label": f"MAE (arXiv {_EX_MAE_ARXIV_ID})",
            "arxiv_id": _EX_MAE_ARXIV_ID,
        },
    }


@app.get("/examples/mae/pdf")
def examples_mae_pdf() -> FileResponse:
    """Serve the anonymized MAE PDF as a blob. The UI fetches this, wraps it
    in a File, and POSTs to /submit -- same code path as a manual upload."""
    if not _EX_MAE_PDF.is_file():
        raise HTTPException(404, f"MAE PDF not present at {_EX_MAE_PDF}")
    return FileResponse(_EX_MAE_PDF, media_type="application/pdf",
                        filename="mae_anonymized.pdf")


@app.get("/examples/mae/latex_path")
def examples_mae_latex_path() -> dict:
    """Return the absolute on-disk path of the bundled MAE LaTeX dir so the
    UI can pass it straight to /submit_latex (same as if the user had
    Browse'd to the folder). We never expose the dir's *contents* over HTTP
    -- /submit_latex hands the path to paperprep, which reads the files
    locally on the server."""
    if not _EX_MAE_LATEX_DIR.is_dir():
        raise HTTPException(404, f"MAE LaTeX dir not present at {_EX_MAE_LATEX_DIR}")
    return {"path": str(_EX_MAE_LATEX_DIR.resolve())}


@app.post("/list_dirs")
def list_dirs(req: ListDirsReq) -> dict:
    """Server-side folder browser: list immediate sub-directories of ``path``.

    Default initial path is the writable per-user scratch dir if present
    (e.g. ``/scratch/gpfs/<group>/<user>``), else ``/scratch/gpfs``, else
    the server user's home -- same logic as the setup wizard's parent-dir
    default so the browser opens where your repos actually live.

    The LaTeX source lives on the *server* filesystem -- paperprep compiles it
    and ``git archive`` reads its history there -- so the UI browses the server
    rather than a browser folder-picker (which can't expose an absolute path).
    Read-only: lists directories, never file contents.
    """
    start = Path(req.path).expanduser() if req.path else _default_browse_path()
    if not start.exists() or not start.is_dir():
        start = _default_browse_path()
    start = start.resolve()
    try:
        entries = sorted(start.iterdir(), key=lambda p: p.name.lower())
    except PermissionError:
        raise HTTPException(403, f"permission denied: {start}")
    dirs: list[dict] = []
    for child in entries:
        if child.name.startswith(".") or not child.is_dir():
            continue
        try:
            is_git = (child / ".git").exists()
            has_tex = any(child.glob("*.tex"))
        except (PermissionError, OSError):
            is_git = has_tex = False
        dirs.append({"name": child.name, "path": str(child),
                     "is_git": is_git, "has_tex": has_tex})
    parent = str(start.parent) if start.parent != start else None
    return {"path": str(start), "parent": parent, "dirs": dirs,
            "bookmarks": _browse_bookmarks()}


@app.post("/probe_latex")
def probe_latex(req: ProbeLatexReq) -> dict:
    """Inspect a local LaTeX dir: validate, suggest the entrypoint, list .tex
    files, and (when git-tracked) return the last-N-commit .tex churn window so
    the UI can render the history graph + pre-select commits above the cutoff.
    """
    p = Path(req.path).expanduser()
    if not p.exists():
        raise HTTPException(400, f"path does not exist: {p}")
    if not p.is_dir():
        raise HTTPException(400, f"not a directory: {p}")
    p = p.resolve()

    info: dict[str, Any] = {
        "ok": True,
        "path": str(p),
        "suggested_main_tex": latexsrc.find_main_tex(p),
        "tex_files": latexsrc.list_tex_files(p),
        "is_git": latexsrc.is_git_repo(p),
    }
    if not info["tex_files"]:
        info["warning"] = "no .tex files found under this directory"
    if info["is_git"]:
        repo = latexsrc.git_toplevel(p) or p
        info["git_toplevel"] = str(repo)
        info["dirty"] = latexsrc.working_tree_dirty(repo)
        try:
            info["history"] = latexsrc.last_commits_with_tex_churn(repo, n=50)
        except Exception as e:
            info["history"] = None
            info["history_error"] = str(e)
    return info


@app.post("/submit_latex")
def submit_latex(req: SubmitLatexReq) -> dict:
    """Kick off a LaTeX-source review. mode=latest scores the working tree as-is
    (one verdict); mode=history scores each selected commit -> a p_accept
    trajectory. Returns the job_id; the UI polls /status/{job_id}.
    """
    p = Path(req.path).expanduser()
    if not p.is_dir():
        raise HTTPException(400, f"not a directory: {req.path}")
    p = p.resolve()
    mode = req.mode if req.mode in ("latest", "history") else "latest"

    if mode == "history":
        if not latexsrc.is_git_repo(p):
            raise HTTPException(400, f"history mode requires a git repo: {p}")
        repo = latexsrc.git_toplevel(p) or p
        commits = req.commits or []
        if not commits:
            # default selection: last-N .tex commits with churn >= the cutoff
            commits = [c for c in latexsrc.last_commits_with_tex_churn(repo, n=req.n_window)["commits"]
                       if c.get("above_pct")]
        # validate / normalize each commit sha
        norm: list[dict] = []
        for c in commits:
            sha = c.get("sha") if isinstance(c, dict) else c
            if not sha:
                continue
            full = latexsrc.resolve_commit(repo, sha)
            if not full:
                raise HTTPException(400, f"unknown commit: {sha!r}")
            norm.append({"sha": full, "short": full[:8],
                         "date": (c.get("date") if isinstance(c, dict) else None),
                         "subject": (c.get("subject") if isinstance(c, dict) else None),
                         "churn": (c.get("churn") if isinstance(c, dict) else None)})
        if not norm:
            raise HTTPException(400, "history mode: no commits selected")

    job_id = uuid.uuid4().hex[:12]
    status = _state["jobs"].create(job_id)
    log.info("[/submit_latex] job=%s path=%s mode=%s main_tex=%s n=%s",
             job_id, p, mode, req.main_tex,
             len(norm) if mode == "history" else 1)

    def _runner():
        with _state["job_lock"]:
            if mode == "history":
                run_latex_history_job(_state["cfg"], status, _state["work_dir"],
                                      p, req.main_tex, norm, max_pages=req.max_pages)
            else:
                run_latex_job(_state["cfg"], status, _state["work_dir"],
                              p, req.main_tex, max_pages=req.max_pages)

    t = threading.Thread(target=_runner, daemon=True, name=f"job-{job_id}")
    t.start()
    return {"job_id": job_id, "stages": STAGES, "submitted_at": status.started_at,
            "mode": mode}


@app.get("/status/{job_id}")
def status_endpoint(job_id: str) -> dict:
    j = _state["jobs"].get(job_id)
    if not j:
        raise HTTPException(404, f"unknown job_id: {job_id!r}")
    return j.to_dict()


@app.get("/jobs")
def list_jobs() -> list[dict]:
    return [j.to_dict() for j in _state["jobs"].all()]


# --------- Per-job file explorer (read-only) ---------

import json as _json
import mimetypes
from pathlib import Path as _Path


_TREE_MAX_FILES_PER_DIR = 500


def _find_trajectory_rec(result: dict, commit: str) -> Optional[dict]:
    """Look up a commit in a latex_history result by short sha or full sha."""
    for rec in (result.get("trajectory") or []):
        if rec.get("short") == commit or rec.get("sha") == commit:
            return rec
    return None


def _job_roots(job_id: str, *, commit: Optional[str] = None) -> dict[str, _Path]:
    """Return {label: root_path} pairs the explorer is allowed to read.

    ``commit`` scopes to a single commit of a latex_history job: the explorer
    sees just that commit's archived source tree + paperprep output, not the
    whole multi-commit job dir. Used by the trajectory drilldown panel so each
    point in the chart resolves to *its* tree/payload.
    """
    j = _state["jobs"].get(job_id)
    if not j:
        raise HTTPException(404, f"unknown job_id: {job_id!r}")
    if not j.result:
        raise HTTPException(409, f"job {job_id} not done yet (state={j.state!r})")
    if commit:
        rec = _find_trajectory_rec(j.result, commit)
        if rec is None:
            raise HTTPException(404, f"commit {commit!r} not in this job's trajectory")
        roots: dict[str, _Path] = {}
        src = rec.get("src_dir")
        if src and _Path(src).exists():
            roots["src"] = _Path(src).resolve()
        pp = rec.get("paperprep_output_dir")
        if pp and _Path(pp).exists():
            roots["paperprep"] = _Path(pp).resolve()
        if not roots:
            raise HTTPException(404, f"no on-disk artifacts left for commit {commit!r}")
        return roots
    roots = {"review": _Path(j.result["job_dir"]).resolve()}
    pp = j.result.get("paperprep_output_dir")
    if pp:
        roots["paperprep"] = _Path(pp).resolve()
    return roots


def _list_tree(root: _Path) -> dict:
    if not root.exists():
        return {"name": root.name, "path": str(root), "type": "missing"}
    if root.is_file():
        return {"name": root.name, "path": str(root), "type": "file",
                "size": root.stat().st_size}
    children: list[dict] = []
    try:
        entries = sorted(root.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    except PermissionError:
        return {"name": root.name, "path": str(root), "type": "dir",
                "children": [], "error": "permission denied"}
    for child in entries:
        if len(children) >= _TREE_MAX_FILES_PER_DIR:
            children.append({"name": "...", "type": "truncated"})
            break
        children.append(_list_tree(child))
    return {"name": root.name, "path": str(root), "type": "dir", "children": children}


@app.get("/jobs/{job_id}/tree")
def job_tree(job_id: str, commit: Optional[str] = None) -> dict:
    return {label: _list_tree(root)
            for label, root in _job_roots(job_id, commit=commit).items()}


@app.get("/jobs/{job_id}/agent_events")
def job_agent_events(job_id: str,
                     variant: str,
                     since: int = 0,
                     limit: int = 500) -> dict:
    """Return the next ``limit`` events from this job's agent transcript,
    starting at sequence ``since``. ``variant`` is ``"no_prior"`` or
    ``"with_prior"``. The UI's chat popup polls this on the same 1.5s tick
    as /status, advancing ``since`` to ``next_since`` each turn.
    """
    j = _state["jobs"].get(job_id)
    if not j:
        raise HTTPException(404, f"unknown job_id: {job_id!r}")
    if variant not in ("no_prior", "with_prior"):
        raise HTTPException(400, f"variant must be no_prior|with_prior, got {variant!r}")
    bucket = j.agent_events.get(variant, [])
    total = len(bucket)
    since = max(0, int(since))
    end = min(total, since + max(1, int(limit)))
    slice_ = bucket[since:end]
    return {
        "job_id": job_id,
        "variant": variant,
        "agent": j.agent_choice,
        "state": j.agent_state.get(variant, "queued"),
        "since": since,
        "next_since": end,
        "total": total,
        "events": slice_,
        "result": j.agent_results.get(variant),
    }


@app.get("/jobs/{job_id}/file")
def job_file(job_id: str, path: str, commit: Optional[str] = None):
    """Stream a file from the job's allowed roots. Rejects path-traversal."""
    target = _Path(path).resolve()
    roots = _job_roots(job_id, commit=commit)
    if not any(target == r or target.is_relative_to(r) for r in roots.values()):
        raise HTTPException(403, f"path not under any allowed root for job {job_id}")
    if not target.is_file():
        raise HTTPException(404, f"not a file: {target}")
    media_type, _ = mimetypes.guess_type(str(target))
    if media_type is None:
        # Default to text/plain for unknown extensions so the browser inlines them
        media_type = "text/plain; charset=utf-8"
    headers = {"Content-Disposition": f'inline; filename="{target.name}"'}
    return FileResponse(target, media_type=media_type, headers=headers)


@app.get("/jobs/{job_id}/payload")
def job_payload(job_id: str, commit: Optional[str] = None) -> dict:
    """Return the exact sharegpt row paperlens-serve scored, plus the score.

    ``commit`` (latex_history only) scopes to that commit's paperprep output
    so each point in the trajectory chart can render the row PaperLens
    actually saw for *that* commit.
    """
    j = _state["jobs"].get(job_id)
    if not j:
        raise HTTPException(404, f"unknown job_id: {job_id!r}")
    if not j.result:
        raise HTTPException(409, f"job {job_id} not done yet")
    modality = j.result.get("modality", "vision")
    if commit:
        rec = _find_trajectory_rec(j.result, commit)
        if rec is None:
            raise HTTPException(404, f"commit {commit!r} not in this job's trajectory")
        pp = rec.get("paperprep_output_dir")
        score: dict = {**rec}
    else:
        pp = j.result.get("paperprep_output_dir")
        score = dict(j.result)
    if not pp:
        raise HTTPException(404, "paperprep_output_dir missing from result")
    sg_path = _Path(pp) / "sharegpt" / modality / "data.json"
    if not sg_path.exists():
        raise HTTPException(404, f"sharegpt export not found: {sg_path}")
    rows = _json.loads(sg_path.read_text())
    row = rows[0] if rows else None
    if row is not None:
        # Mirror what pipeline.py appends before POSTing to paperlens-serve.
        convs = list(row.get("conversations", []))
        if not any(c.get("from") == "gpt" for c in convs):
            convs.append({"from": "gpt", "value": "Outcome: \\boxed{Accept}"})
            row = {**row, "conversations": convs}
    return {
        "job_id": job_id,
        "commit": commit,
        "modality": modality,
        "sharegpt_path": str(sg_path),
        "row": row,
        "score": score,
    }


# --------- Entry point ---------

def run_uvicorn(cfg_path: str, host: Optional[str] = None, port: Optional[int] = None) -> int:
    import re
    import uvicorn
    cfg = OmegaConf.load(cfg_path)
    host = host or cfg.server.host
    port = port or int(cfg.server.port)
    os.environ["PAPERLENSREVIEW_CONFIG"] = cfg_path
    log_level = str(cfg.get("logging", {}).get("level", "INFO")).lower()
    logging.basicConfig(level=log_level.upper(),
                        format="%(asctime)s %(name)s %(levelname)s: %(message)s")

    # Browsers poll /status/<jobid> every 1.5s while a job runs, so a single
    # latex_history job emits ~40 access lines per minute and buries the
    # actually-interesting log events. Drop those (and per-job tree/file/payload
    # GETs, which fire repeatedly when the file-explorer panel is open) from
    # the uvicorn access log. Real submits, /health, and the initial GET / are
    # all kept.
    _NOISE_RE = re.compile(
        r' "GET /(?:status/[^"]*|jobs/[^"]+/(?:tree|file|payload)[^"]*) HTTP/'
    )

    class _DropPollNoise(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:  # True = keep
            return _NOISE_RE.search(record.getMessage()) is None

    logging.getLogger("uvicorn.access").addFilter(_DropPollNoise())

    log.info("paperlensreview binding %s:%s", host, port)
    uvicorn.run(app, host=host, port=port, log_level=log_level)
    return 0
