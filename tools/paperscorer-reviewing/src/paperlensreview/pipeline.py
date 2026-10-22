"""Per-PDF pipeline: paperprep serve /prepare -> paperlens serve /score.

One job_id -> one subdir under ``cfg.paperprep_serve.work_dir/<job_id>/``. Inside:

  <job_id>.pdf       raw upload (path we hand to paperprep over the wire)
  result.json        final verdict ({decision, p_accept, ...})

The paperprep serve daemon owns its own output_dir (per-request subdirs hold
the compile/mineru/normalize/filter/export artifacts). We don't touch those;
we read the absolute ``sharegpt_vision_path`` / ``sharegpt_text_path`` it
returns and POST a single row to paperlens serve /score.

Both upstream services are persistent FastAPI/Flask daemons -- launch_local.sh
brings them up and waits for /healthz before exec'ing the reviewing server.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import requests


log = logging.getLogger(__name__)


# Two coarse stages: paperprep serve runs synchronously and only returns when
# its internal compile/mineru/normalize/filter/export chain is done, so we
# can't show fine-grained progress for it without modifying paperprep itself.
STAGES: list[str] = ["paperprep", "paperlens_score"]

# Appended when the user opted into the post-decision agentic review on the
# PDF / arxiv tabs (claude or codex headless). agent_setup builds the two
# workspaces (no-prior + with-prior); agent_review streams both runs in
# parallel through the chat popups.
STAGES_AGENT: list[str] = ["agent_setup", "agent_review"]


@dataclass
class JobStatus:
    job_id: str
    state: str = "queued"               # queued | running | done | error
    stage: Optional[str] = None         # one of `stages` (or None before start / "done")
    stage_index: int = -1               # 0..len(stages)-1 (or -1 / len(stages))
    total_stages: int = len(STAGES)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    error: Optional[str] = None
    result: Optional[dict] = None       # final decision (decision, p_accept, ...)
    stage_log: list[dict] = field(default_factory=list)   # one entry per stage transition
    # Per-job stage list -- defaults to the standard 2-stage PDF/LaTeX path.
    # The arxiv flow prepends "download" + "latex_extraction", and the history
    # flow leaves this alone and just overrides total_stages with the commit
    # count. _enter_stage uses this list to compute stage_index correctly.
    stages: list[str] = field(default_factory=lambda: list(STAGES))

    # ----- Agentic review state (PDF + arxiv tabs only) ---------------------
    # When the user toggles "Run agentic review", these track the two parallel
    # subagent runs (no-prior + with-prior). Events accumulate per-variant so
    # the UI can poll /jobs/<id>/agent_events?variant=... and render a chat
    # transcript that streams in. ``agent_choice`` is "claude" | "codex" | None.
    agent_choice: Optional[str] = None
    agent_state: dict = field(default_factory=dict)   # {variant: "queued|running|done|error"}
    agent_events: dict = field(default_factory=dict)  # {variant: list[event dict]}
    agent_results: dict = field(default_factory=dict) # {variant: {predictions, error?, workspace, elapsed_s}}

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "state": self.state,
            "stage": self.stage,
            "stage_index": self.stage_index,
            "total_stages": self.total_stages,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "result": self.result,
            "stage_log": list(self.stage_log),
            "stages": list(self.stages),
            "agent_choice": self.agent_choice,
            # Don't ship full event arrays through /status -- the UI fetches
            # them lazily from /jobs/<id>/agent_events. Just expose counts +
            # state so the popup can show "12 events, running" without bloat.
            "agent_state": dict(self.agent_state),
            "agent_event_counts": {k: len(v) for k, v in self.agent_events.items()},
            "agent_results": {k: {kk: vv for kk, vv in r.items() if kk != "predictions"}
                              for k, r in self.agent_results.items()},
        }


class JobRegistry:
    """In-memory store of JobStatus, keyed by job_id."""

    def __init__(self):
        self._jobs: dict[str, JobStatus] = {}
        self._lock = threading.Lock()

    def create(self, job_id: str) -> JobStatus:
        with self._lock:
            s = JobStatus(job_id=job_id)
            self._jobs[job_id] = s
            return s

    def get(self, job_id: str) -> Optional[JobStatus]:
        with self._lock:
            return self._jobs.get(job_id)

    def all(self) -> list[JobStatus]:
        with self._lock:
            return list(self._jobs.values())


# ---------------------------------------------------------------------------
# Stage transition helper
# ---------------------------------------------------------------------------

def _enter_stage(status: JobStatus, stage: str, via: str = "transition") -> None:
    status.stage = stage
    try:
        status.stage_index = status.stages.index(stage)
    except ValueError:
        status.stage_index = len(status.stages)
    status.stage_log.append({"t": time.time(), "stage": stage, "via": via})


# ---------------------------------------------------------------------------
# paperprep serve client
# ---------------------------------------------------------------------------

def _call_paperprep_prepare(cfg, request_id: str, *, input_type: str,
                            path: Path, main_tex: Optional[str] = None,
                            max_pages: Optional[int] = None) -> dict:
    """POST one paper (pdf or latex_dir) to paperprep serve /prepare.

    paperprep serve writes its outputs under its own output_dir (configured at
    daemon launch). The response has absolute paths to the ShareGPT exports we
    then forward to paperlens-serve. ``main_tex`` is an optional entrypoint hint
    for latex_dir (paperprep auto-detects when omitted). ``max_pages`` overrides
    paperprep's server-level page cap for this request (e.g. score the first 16
    pages instead of the default 14); None falls back to the server default.
    """
    url = cfg.paperprep_serve.base_url.rstrip("/") + "/prepare"
    item = {"id": request_id, "type": input_type, "path": str(path)}
    if input_type == "latex_dir" and main_tex:
        item["main_tex"] = main_tex
    if max_pages is not None:
        item["max_pages"] = int(max_pages)
    payload = {"request_id": request_id, "papers": [item]}
    timeout = float(cfg.paperprep_serve.timeout_seconds)
    r = requests.post(url, json=payload, timeout=timeout)
    if r.status_code >= 500:
        raise RuntimeError(f"paperprep serve {r.status_code}: {r.text[:400]}")
    r.raise_for_status()
    body = r.json()
    if not isinstance(body, dict):
        raise RuntimeError(f"paperprep serve returned non-dict body: {body!r}")
    return body


def _load_sharegpt_export(prepare_body: dict, modality: str) -> Optional[dict]:
    """Resolve which sharegpt_{modality}_path to read from paperprep's response,
    load the first row, and append a placeholder gpt turn so LF's 'ppo' stage
    accepts it at inference time. Returns None on any miss.
    """
    key = f"sharegpt_{modality}_path"
    sg_path_str = prepare_body.get(key)
    if not sg_path_str:
        log.warning("paperprep prepare body missing %s: %r", key, prepare_body)
        return None
    sg_path = Path(sg_path_str)
    if not sg_path.exists():
        log.warning("paperprep sharegpt export path doesn't exist: %s", sg_path)
        return None
    try:
        rows = json.loads(sg_path.read_text())
    except Exception as e:
        log.warning("paperprep sharegpt parse failed (%s): %s", sg_path, e)
        return None
    if not rows:
        return None
    # paperprep emits inference-only rows ([system, human], no gpt). The
    # paperlens-serve /score endpoint goes through LF's "ppo" stage which
    # expects an assistant turn even at inference time. Append a neutral
    # placeholder; it's discarded after tokenization.
    row = dict(rows[0])
    convs = list(row.get("conversations", []))
    if not any(c.get("from") == "gpt" for c in convs):
        convs.append({"from": "gpt", "value": "Outcome: \\boxed{Accept}"})
    row["conversations"] = convs
    return row


# ---------------------------------------------------------------------------
# paperlens serve client
# ---------------------------------------------------------------------------

def _score_via_paperlens(cfg, sharegpt_row: dict) -> dict:
    """POST one sharegpt row to paperlens serve /score, return its
    {p_accept, logp_accept, logp_reject, pred} dict.
    """
    url = cfg.paperlens_serve.base_url.rstrip("/") + "/score"
    payload = {"papers": [sharegpt_row]}
    r = requests.post(url, json=payload, timeout=float(cfg.paperlens_serve.timeout_seconds))
    r.raise_for_status()
    body = r.json()
    if not body.get("scores"):
        raise RuntimeError(f"paperlens-serve returned no scores: {body}")
    return body["scores"][0]


def _decision_from_p_accept(p_accept: float, threshold: float = 0.5) -> str:
    return "Accept" if p_accept >= threshold else "Reject"


# ---------------------------------------------------------------------------
# Main job entry point
# ---------------------------------------------------------------------------

def _score_from_prepare(cfg, prepare_body: dict) -> dict:
    """Validate the per-paper prepare result, load its sharegpt row, score it.

    Returns the score-derived fields; raises on any failure. Shared by the PDF,
    single-latex, and per-commit history paths.
    """
    papers = prepare_body.get("papers") or []
    if not papers:
        raise RuntimeError(f"paperprep prepare returned no papers: {prepare_body}")
    p0 = papers[0]
    if p0.get("status") != "ok":
        raise RuntimeError(
            f"paperprep failed at paper level: status={p0.get('status')!r} "
            f"error={p0.get('error')!r}"
        )
    modality = str(cfg.review.modality)
    row = _load_sharegpt_export(prepare_body, modality)
    if row is None:
        raise RuntimeError(
            f"paperprep prepare did not produce a {modality} sharegpt row; "
            f"response keys={list(prepare_body.keys())}"
        )
    score = _score_via_paperlens(cfg, row)

    # Build a 5x2 panel preview from the first 10 pages -- the trajectory
    # drilldown swaps these in when you click a commit point. Best-effort,
    # never blocks scoring (returns None if Pillow missing or page IO fails).
    panel_path: Optional[str] = None
    pp = prepare_body.get("output_dir")
    if pp and row.get("images"):
        from . import panel as _panel
        out = _panel.build_panel([Path(p) for p in row["images"]],
                                 Path(pp) / "panel.png")
        if out is not None:
            panel_path = str(out)

    pa = float(score["p_accept"])
    return {
        "p_accept": pa,
        "decision": _decision_from_p_accept(pa),
        "logp_accept": score.get("logp_accept"),
        "logp_reject": score.get("logp_reject"),
        "pred": score.get("pred"),
        "body_pages": p0.get("body_pages"),
        "paperprep_output_dir": prepare_body.get("output_dir"),
        "paperprep_elapsed_s": prepare_body.get("elapsed_s"),
        "panel_path": panel_path,
    }


def _write_done(status: JobStatus, job_dir: Path, result: dict) -> None:
    """Mark the job's primary result and (when there's no agent stage) flip
    state to "done". When an agent stage is queued we keep state="running"
    so the UI keeps polling while the agent transcripts stream in -- the
    result.json is still written immediately so the verdict shows up before
    the agents finish.
    """
    (job_dir / "result.json").write_text(json.dumps(result, indent=2))
    status.result = result
    if status.agent_choice:
        # Agent stage is up next; advance to it and stay running.
        status.stage = "agent_setup"
        try:
            status.stage_index = status.stages.index("agent_setup")
        except ValueError:
            status.stage_index = len(STAGES)
    else:
        status.state = "done"
        status.stage = "done"
        status.stage_index = status.total_stages


# ---------------------------------------------------------------------------
# Agent stage (post-decision, optional)
# ---------------------------------------------------------------------------

_AGENT_VARIANTS = ("no_prior", "with_prior")


def _agent_text_md(prepare_body: dict) -> Optional[Path]:
    """Find paperprep's normalized text.md for this paper. We look under the
    same output_dir tree the panel builder pulls page_images from."""
    pp = prepare_body.get("output_dir")
    if not pp:
        return None
    for cand in Path(pp).rglob("text.md"):
        return cand
    for cand in Path(pp).rglob("*.md"):
        return cand
    return None


def _run_agent_stage(status: JobStatus, job_dir: Path, *, result: dict,
                     prepare_body: dict) -> None:
    """Build two workspaces (no-prior + with-prior) and run the chosen agent
    in each concurrently. Streams normalized events into ``status.agent_events``;
    the UI polls /jobs/<id>/agent_events?variant=... to render them as chat.
    """
    from . import agents as _agents

    agent = status.agent_choice or ""
    job_id = status.job_id
    pp_dir = Path(prepare_body.get("output_dir") or job_dir)
    panel = result.get("panel_path")
    panel_path = Path(panel) if panel else None
    text_md = _agent_text_md(prepare_body)
    sid = result.get("arxiv_id") or job_id
    decision = result.get("decision", "Accept")
    p_accept = float(result.get("p_accept", 0.5))
    title = result.get("title") or result.get("pdf_name") or sid
    modality = str(result.get("modality", "vision"))
    domain = str(result.get("domain", ""))

    # Compute the calibrated PaperLens prior ONCE -- both variants share the
    # same Platt-scaled bundle (with_prior shows it; no_prior ignores it).
    # The score row in `result` carries the raw 2-token logprobs which is the
    # cleanest input for Platt (logit difference = z directly).
    cal = _agents.calibrate_prior(
        logp_accept=result.get("logp_accept"),
        logp_reject=result.get("logp_reject"),
        p_accept_raw=p_accept,
        domain=domain,
        modality=modality,
    )
    log.info("[%s] calibrated prior: domain=%s modality=%s -> %s",
             job_id, domain, modality, {k: cal.get(k) for k in
             ("p_accept_raw", "p_accept_cal", "confidence", "decision", "has_calibration")})

    # Stage 1: agent_setup -- build both workspaces.
    _enter_stage(status, "agent_setup", via="agent")
    manifests: dict[str, dict] = {}
    setup_root = job_dir / "agents" / agent
    for variant in _AGENT_VARIANTS:
        ws = setup_root / variant
        manifests[variant] = _agents.build_workspace(
            dest=ws,
            paperprep_output_dir=pp_dir,
            submission_id=str(sid),
            modality=modality,
            panel_path=panel_path,
            text_md_path=text_md,
            paper_title=str(title),
            with_prior=(variant == "with_prior"),
            prior_decision=decision,
            prior_p_accept=p_accept,
            prior_calibration=cal,
            domain=domain,
            agent=agent,
        )
        status.agent_state[variant] = "queued"
        status.agent_events[variant] = []
    # Surface the calibration bundle on result.json so the UI's verdict card
    # can show "raw 0.881 / calibrated 0.811 / confidence 0.811" alongside
    # the agent transcripts.
    if status.result is not None:
        status.result["paperlens_prior"] = cal
    log.info("[%s] agent workspaces built -> %s/{no_prior,with_prior}", job_id, setup_root)

    # Stage 2: agent_review -- spawn both runs in their own threads.
    _enter_stage(status, "agent_review", via="agent")

    def _append(variant: str, ev_kind_payload: dict) -> None:
        bucket = status.agent_events.setdefault(variant, [])
        bucket.append({
            "seq": len(bucket),
            "ts": time.time(),
            "kind": ev_kind_payload.get("kind", "status"),
            "payload": ev_kind_payload.get("payload", {}),
        })

    threads: list[threading.Thread] = []
    runner_lock = threading.Lock()
    runners_done: dict[str, _agents.AgentResult] = {}

    def _runner(variant: str, manifest: dict) -> None:
        status.agent_state[variant] = "running"
        try:
            res = _agents.run_agent_headless(
                agent=agent,
                workspace=Path(manifest["workspace"]),
                prompt=manifest["prompt"],
                submission_id=str(sid),
                on_event=lambda ev: _append(variant, ev),
            )
        except Exception as e:
            log.exception("[%s] agent %s/%s crashed", job_id, agent, variant)
            _append(variant, {"kind": "error", "payload": {"msg": f"runner crashed: {e}"}})
            status.agent_state[variant] = "error"
            with runner_lock:
                runners_done[variant] = _agents.AgentResult(
                    ok=False, exit_code=-1, final_text="", error=str(e),
                    workspace=manifest["workspace"])
            return
        status.agent_state[variant] = "done" if res.ok else "error"
        with runner_lock:
            runners_done[variant] = res

    for variant in _AGENT_VARIANTS:
        t = threading.Thread(target=_runner, args=(variant, manifests[variant]),
                             daemon=True, name=f"agent-{agent}-{variant}-{job_id[:6]}")
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    # Surface predictions + workspace paths on status.result so the UI's
    # final card can show "agent decision vs paperlens decision".
    agent_summary: dict = {}
    for variant in _AGENT_VARIANTS:
        r = runners_done.get(variant)
        agent_summary[variant] = {
            "workspace": (r.workspace if r else None),
            "predictions": (r.predictions if r else None),
            "ok": bool(r and r.ok),
            "error": (r.error if r else "no result"),
            "exit_code": (r.exit_code if r else -1),
        }
        status.agent_results[variant] = agent_summary[variant]

    if status.result is not None:
        status.result["agent"] = {"choice": agent, **agent_summary}
        (job_dir / "result.json").write_text(json.dumps(status.result, indent=2))

    # Final: agents done, mark the job complete.
    status.state = "done"
    status.stage = "done"
    status.stage_index = status.total_stages


def run_job(cfg, status: JobStatus, work_dir: Path, pdf_bytes: bytes, pdf_name: str,
            max_pages: Optional[int] = None) -> None:
    """End-to-end PDF job. Writes ``result.json`` on success; sets ``status.error``
    on failure. Designed to be called inside a thread. ``max_pages`` overrides
    paperprep's page cap for this job (None = server default).
    """
    job_id = status.job_id
    job_dir = work_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = job_dir / f"{job_id}.pdf"
    pdf_path.write_bytes(pdf_bytes)
    log.info(f"[{job_id}] pdf saved -> {pdf_path} ({len(pdf_bytes)} bytes)")

    status.state = "running"
    status.started_at = time.time()
    _enter_stage(status, "paperprep")
    try:
        log.info(f"[{job_id}] paperprep /prepare (pdf={pdf_path}, max_pages={max_pages})")
        prepare_body = _call_paperprep_prepare(cfg, job_id, input_type="pdf", path=pdf_path,
                                               max_pages=max_pages)
        _enter_stage(status, "paperlens_score")
        sc = _score_from_prepare(cfg, prepare_body)
        result = {
            **sc,
            "modality": str(cfg.review.modality),
            "domain": str(cfg.review.domain),
            "source_type": "pdf",
            "pdf_name": pdf_name,
            "job_dir": str(job_dir),
        }
        _write_done(status, job_dir, result)
        log.info(f"[{job_id}] result: {result['decision']} (p_accept={result['p_accept']:.4f})")
        if status.agent_choice:
            _run_agent_stage(status, job_dir, result=result, prepare_body=prepare_body)
    except Exception as e:
        log.exception(f"[{job_id}] pipeline failed")
        status.state = "error"
        status.error = str(e)
    finally:
        status.finished_at = time.time()


def run_latex_job(cfg, status: JobStatus, work_dir: Path, src_path: Path,
                  main_tex: Optional[str], max_pages: Optional[int] = None) -> None:
    """Review a LaTeX source dir as-is (working tree). Single verdict, mirrors
    the PDF flow but with type=latex_dir so paperprep compiles + anonymizes.
    ``max_pages`` overrides paperprep's page cap (None = server default).
    """
    job_id = status.job_id
    job_dir = work_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    status.state = "running"
    status.started_at = time.time()
    _enter_stage(status, "paperprep")
    try:
        log.info(f"[{job_id}] paperprep /prepare (latex_dir={src_path}, main_tex={main_tex}, max_pages={max_pages})")
        prepare_body = _call_paperprep_prepare(
            cfg, job_id, input_type="latex_dir", path=src_path, main_tex=main_tex or None,
            max_pages=max_pages)
        _enter_stage(status, "paperlens_score")
        sc = _score_from_prepare(cfg, prepare_body)
        result = {
            **sc,
            "modality": str(cfg.review.modality),
            "domain": str(cfg.review.domain),
            "source_type": "latex_dir",
            "latex_dir": str(src_path),
            "main_tex": main_tex or None,
            "git_mode": "latest",
            "job_dir": str(job_dir),
        }
        _write_done(status, job_dir, result)
        log.info(f"[{job_id}] result: {result['decision']} (p_accept={result['p_accept']:.4f})")
        if status.agent_choice:
            _run_agent_stage(status, job_dir, result=result, prepare_body=prepare_body)
    except Exception as e:
        log.exception(f"[{job_id}] latex pipeline failed")
        status.state = "error"
        status.error = str(e)
    finally:
        status.finished_at = time.time()


def run_latex_history_job(cfg, status: JobStatus, work_dir: Path, repo_path: Path,
                          main_tex: Optional[str], commits: list[dict],
                          max_pages: Optional[int] = None) -> None:
    """Score a paper across selected git commits -> a p_accept trajectory.

    Each commit is git-archived into ``<job_dir>/src/<short>``, prepped + scored.
    Per-commit paperprep artifacts (source tree + MinerU outputs + sharegpt
    export) are KEPT so the UI's drilldown panel can render that commit's
    Files browser and the exact sharegpt row PaperLens scored on. Inode cost:
    K commits x a few hundred MinerU crops each -- run ``checkquota`` if you
    chain very long histories (the ZHUANGL group is at ~175M cap).

    Per-commit failures are recorded in the trajectory and not fatal.
    """
    from . import latexsrc

    job_id = status.job_id
    job_dir = work_dir / job_id
    src_root = job_dir / "src"
    src_root.mkdir(parents=True, exist_ok=True)

    status.state = "running"
    status.started_at = time.time()
    status.total_stages = max(1, len(commits))
    repo = latexsrc.git_toplevel(repo_path) or repo_path

    trajectory: list[dict] = []
    try:
        for i, c in enumerate(commits):
            sha = c["sha"]
            short = c.get("short") or sha[:8]
            status.stage = f"commit {i + 1}/{len(commits)}: {short}"
            status.stage_index = i
            status.stage_log.append({"t": time.time(), "stage": status.stage, "via": "commit"})
            rec = {"order": i, "sha": sha, "short": short,
                   "date": c.get("date"), "subject": c.get("subject"),
                   "churn": c.get("churn"), "state": "pending"}
            try:
                tree = src_root / short
                latexsrc.archive_commit(repo, sha, tree)
                body = _call_paperprep_prepare(
                    cfg, f"{job_id}_{short}", input_type="latex_dir",
                    path=tree, main_tex=main_tex or None, max_pages=max_pages)
                sc = _score_from_prepare(cfg, body)
                rec.update({
                    "state": "done",
                    "p_accept": round(sc["p_accept"], 4),
                    "decision": sc["decision"],
                    "logp_accept": sc.get("logp_accept"),
                    "logp_reject": sc.get("logp_reject"),
                    "body_pages": sc.get("body_pages"),
                    # Per-commit artifacts kept around for the UI drilldown.
                    # /jobs/<id>/{tree,file,payload}?commit=<short> scopes
                    # to these dirs via _job_roots(commit=...).
                    "paperprep_output_dir": sc.get("paperprep_output_dir"),
                    "src_dir": str(tree),
                    # 5x2 panel preview the trajectory's right rail swaps in.
                    "panel_path": sc.get("panel_path"),
                })
                log.info(f"[{job_id}] commit {short}: {rec['decision']} p_accept={rec['p_accept']}")
            except Exception as e:
                rec.update({"state": "error", "error": str(e)})
                log.warning(f"[{job_id}] commit {short} failed: {e}")
            trajectory.append(rec)

        result = {
            "source_type": "latex_history",
            "modality": str(cfg.review.modality),
            "domain": str(cfg.review.domain),
            "repo": str(repo),
            "main_tex": main_tex or None,
            "git_mode": "history",
            "n_commits": len(commits),
            "n_scored": sum(1 for r in trajectory if r["state"] == "done"),
            "trajectory": trajectory,
            "job_dir": str(job_dir),
        }
        _write_done(status, job_dir, result)
        log.info(f"[{job_id}] history done: {result['n_scored']}/{len(commits)} commits scored")
    except Exception as e:
        log.exception(f"[{job_id}] latex history job failed")
        status.state = "error"
        status.error = str(e)
    finally:
        status.finished_at = time.time()
