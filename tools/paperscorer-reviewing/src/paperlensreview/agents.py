"""Headless Claude Code / Codex CLI runners for the post-decision agentic
review step.

Calibrated prior: the with-prior workspace injects the Platt-scaled
p_accept (alongside the raw 2-token-renormalized value) so the agent sees
the same calibrated confidence the paper reports. Platt params per
(source, modality) come from PaperLensArXivRelease tab:platt_params (S.G.3
Calibration Details).

Mirrors the paper's Appendix "Agentic Reviewing" scaffold (PaperLensArXivRelease/
main.tex S.appendix:claude_review_prompts) but adapted to a single-paper UI flow:
for each agent run we spin up TWO workspaces (no-prior / with-prior) so the user
sees the calibrated PaperLens prior's effect side-by-side.

Each workspace is self-contained -- text.md, page_images/, panel.png, the schema
validator, and CLAUDE.md or AGENTS.md carrying the appendix prompt. The agent
writes its review to ``agent_reviews/<sid>.json`` and we then collapse it into
the workspace-level ``PREDICTIONS.json`` for the UI to surface.

Streaming: both CLIs emit NDJSON to stdout. We translate each line into a
normalized event ``{type, ts, payload}`` and push it through a callback. The
server keeps a ring buffer per (job_id, variant) so the UI's chat popup can
poll ``/jobs/<id>/agent_events?variant=...&since=...`` and render tool calls
as they happen.

No internet / MCP: the appendix uses an arxiv-search MCP for novelty checks
but the paper's own footnote disables web tools for the deployment; we follow
that and run the agent fully siloed in its workspace.
"""
from __future__ import annotations

import json
import logging
import math
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Platt-scaling calibration (PaperLensArXivRelease tab:platt_params, S.G.3)
# ---------------------------------------------------------------------------
#
# The SFT model emits next-token logprobs at the decision position. Restrict
# to the 2-token decision alphabet {Accept, Reject} and let l_A, l_R be the
# raw logprobs. Then:
#   z       = l_A - l_R          # log-odds of accept
#   p_raw   = sigmoid(z)         # = exp(l_A) / (exp(l_A) + exp(l_R))
#   p_cal   = sigmoid(a*z + b)   # Platt-scaled
#   conf    = p_cal if predicted Accept else 1 - p_cal
#
# a < 1 softens overconfident logits; positive b nudges toward accept (fixes
# over-reject bias on validation). The four (source, modality) cells the
# paper publishes are below; we pick the row from cfg.review.{domain,modality}.
#
# Confidence tracks accuracy near 1:1 in the paper's calibration plots --
# we surface that to the agent in the with-prior prompt so it weights the
# prior by `model_confidence` as a Bayesian prior.

PLATT_PARAMS: dict[tuple[str, str], tuple[float, float]] = {
    ("iclr",  "text"):   (0.897, 0.242),   # OR-ICLR PaperLens-T
    ("iclr",  "vision"): (0.642, 0.174),   # OR-ICLR PaperLens-V
    ("arxiv", "text"):   (0.349, 0.286),   # arXiv    PaperLens-T
    ("arxiv", "vision"): (0.566, 0.047),   # arXiv    PaperLens-V
}


def _platt_for(domain: str, modality: str) -> Optional[tuple[float, float]]:
    return PLATT_PARAMS.get((str(domain).lower(), str(modality).lower()))


def _sigmoid(x: float) -> float:
    if x >= 0:
        ez = math.exp(-x)
        return 1.0 / (1.0 + ez)
    ez = math.exp(x)
    return ez / (1.0 + ez)


def calibrate_prior(*, logp_accept: Optional[float], logp_reject: Optional[float],
                    p_accept_raw: Optional[float],
                    domain: str, modality: str) -> dict:
    """Compute a calibrated prior bundle from paperlens-serve's score row.

    Inputs (any can be missing):
      - logp_accept, logp_reject: raw logprobs of the 2 decision tokens
        (preferred -- yields z directly and matches the paper's procedure)
      - p_accept_raw: server's already-renormalized p_accept; used only as a
        fallback when logprobs are absent

    Returns ``{p_accept_raw, p_accept_cal, z, platt: (a,b), decision,
    confidence, has_calibration}``. ``has_calibration`` is False if we
    couldn't find a Platt row for the (domain, modality) cell.
    """
    z: Optional[float] = None
    p_raw: Optional[float] = None
    if logp_accept is not None and logp_reject is not None:
        z = float(logp_accept) - float(logp_reject)
        p_raw = _sigmoid(z)
    elif p_accept_raw is not None:
        p_raw = float(p_accept_raw)
        # Clamp away from 0/1 before logit so finite z stays finite.
        pc = min(max(p_raw, 1e-6), 1.0 - 1e-6)
        z = math.log(pc / (1.0 - pc))
    out: dict = {
        "p_accept_raw": (round(p_raw, 6) if p_raw is not None else None),
        "z": (round(z, 6) if z is not None else None),
        "has_calibration": False,
    }
    ab = _platt_for(domain, modality)
    if ab is not None and z is not None:
        a, b = ab
        p_cal = _sigmoid(a * z + b)
        decision = "accept" if p_cal >= 0.5 else "reject"
        out.update({
            "platt_a": a,
            "platt_b": b,
            "p_accept_cal": round(p_cal, 6),
            "decision": decision,
            "confidence": round(p_cal if decision == "accept" else 1.0 - p_cal, 6),
            "has_calibration": True,
        })
    else:
        # No Platt row -> fall back to the raw renormalized p_accept (still
        # well-defined; just not Platt-corrected).
        if p_raw is not None:
            decision = "accept" if p_raw >= 0.5 else "reject"
            out.update({
                "decision": decision,
                "confidence": round(p_raw if decision == "accept" else 1.0 - p_raw, 6),
            })
    return out


# ---------------------------------------------------------------------------
# Probe: which agents are available on this host?
# ---------------------------------------------------------------------------

_PROBE_CACHE: Optional[dict] = None
_PROBE_LOCK = threading.Lock()


def probe_agents(refresh: bool = False) -> dict:
    """Return ``{claude: {available, version?}, codex: {...}}``.

    Cached for the process lifetime so the UI's /health poll is cheap.
    ``refresh=True`` forces re-probing (e.g. if the operator just `pip install`'d).
    """
    global _PROBE_CACHE
    with _PROBE_LOCK:
        if _PROBE_CACHE is not None and not refresh:
            return _PROBE_CACHE
        out: dict = {}
        for name, bin_name in (("claude", "claude"), ("codex", "codex")):
            path = shutil.which(bin_name)
            entry: dict = {"available": False, "path": path}
            if path:
                try:
                    r = subprocess.run([path, "--version"],
                                       capture_output=True, text=True, timeout=8)
                    if r.returncode == 0:
                        entry["available"] = True
                        entry["version"] = (r.stdout or r.stderr).strip().splitlines()[0][:120]
                    else:
                        entry["error"] = f"--version exit {r.returncode}"
                except Exception as e:
                    entry["error"] = str(e)[:200]
            out[name] = entry
        _PROBE_CACHE = out
        return out


# ---------------------------------------------------------------------------
# Prompts (verbatim from main.tex appendix, with <SID> substituted)
# ---------------------------------------------------------------------------

_PROMPT_NO_PRIOR = """\
You are {persona}. Review the paper with submission_id={sid}.{venue_note}

STEPS:
1. Read the paper content from papers/{sid}/content.json
2. Examine the page images listed in the content JSON under img_pages
3. Write a structured {review_kind} review as JSON with all 11 fields:
   - decision: "accept" or "reject"
   - rating: 0-10
   - confidence: 1-5
   - soundness: 1-4
   - presentation: 1-4
   - contribution: 1-4
   - summary: 2-3 sentence summary
   - strengths: numbered list citing specific sections/figures/theorems
   - weaknesses: numbered list citing specifics (no generic criticisms)
   - questions: numbered list of questions for authors
   - missing_references: prior work explicitly mentioned in the paper

4. Write the review JSON to agent_reviews/{sid}.json using the Write tool.
5. Run: python check_predictions_schema.py agent_reviews/{sid}.json
6. Return ONLY "Done: {sid}" -- do NOT return the full review text.
"""

_PROMPT_WITH_PRIOR = """\
You are {persona}. Review the paper with submission_id={sid}.{venue_note}

PAPERLENS PRIOR (model: PaperLens-{modality_letter} trained on {domain}):
- Model prediction:                 {decision}
- Raw p(accept)  [2-token softmax]: {p_raw:.3f}
- Calibrated p(accept) [Platt]:     {p_cal_disp}
- Model confidence in its decision: {confidence_disp}

In the paper's calibration plots, PaperLens confidence tracks accuracy
roughly 1:1 -- i.e. a 0.80-confidence prediction is right about 80% of
the time. So high-confidence priors should be trusted strongly; low-
confidence priors leave more room for your own judgment.

STEPS:
1. Read the paper content from papers/{sid}/content.json
2. Examine the page images listed in the content JSON under img_pages
3. Use the model prediction as a Bayesian prior -- weight it by the
   calibrated confidence (which tracks accuracy near 1:1 in the paper's
   data). High confidence -> trust the prior strongly; low confidence
   -> rely on your own reading of the paper.
4. Write a structured {review_kind} review as JSON with all 11 fields:
   - decision: "accept" or "reject"
   - rating: 0-10
   - confidence: 1-5
   - soundness: 1-4
   - presentation: 1-4
   - contribution: 1-4
   - summary: 2-3 sentence summary
   - strengths: numbered list citing specific sections/figures/theorems
   - weaknesses: numbered list citing specifics (no generic criticisms)
   - questions: numbered list of questions for authors
   - missing_references: prior work explicitly mentioned in the paper

5. Write the review JSON to agent_reviews/{sid}.json using the Write tool.
6. Run: python check_predictions_schema.py agent_reviews/{sid}.json
7. Return ONLY "Done: {sid}" -- do NOT return the full review text.
"""

def _persona_for(domain: Optional[str]) -> dict:
    """Return persona+venue strings keyed off the model's training domain.

    ``iclr`` -> "ICLR reviewer" (the calibration anchor is ICLR specifically).
    ``arxiv`` -> a general ML researcher/reviewer; the paper was submitted to
    an unspecified ML venue. Keep the 11-field schema either way -- it's the
    cceval scaffold's review shape, not an ICLR-only contract.
    """
    d = (domain or "").lower()
    if d == "iclr":
        return {"persona": "an ICLR reviewer",
                "venue_note": "",
                "review_kind": "ICLR"}
    if d == "arxiv":
        return {"persona": "an experienced ML researcher and reviewer",
                "venue_note": " It was submitted to an ML venue.",
                "review_kind": "ML conference"}
    # Unknown / future cells -- safe default that doesn't lie about the venue.
    return {"persona": "an experienced ML researcher and reviewer",
            "venue_note": "",
            "review_kind": "ML conference"}


_SYSTEM_RULES_BASE = """\
# {title}

## Output Schema -- agent_reviews/<submission_id>.json

A single JSON object with these 11 mandatory fields:

  - decision: "accept" or "reject"  (lowercase)
  - rating: integer 0-10
  - confidence: integer 1-5
  - soundness: integer 1-4
  - presentation: integer 1-4
  - contribution: integer 1-4
  - summary: 2-3 sentence summary
  - strengths: numbered list citing specific sections/figures/theorems
  - weaknesses: numbered list citing specifics (no generic criticisms)
  - questions: numbered list of questions for authors
  - missing_references: prior work explicitly mentioned in the paper

## Rules

1. Review using BOTH the text content (papers/<sid>/content.json field
   `content_text`, mirrored from text.md) AND the rendered page images
   listed under `img_pages`.
2. Strengths and weaknesses MUST cite specific paper content -- sections,
   theorems, tables, figures. Generic criticisms are prohibited.
3. All 11 fields are mandatory. Decisions must be exactly "accept" or
   "reject" (lowercase). Numeric scores must be in the valid ranges above.
4. Do NOT cd outside the workspace. Use relative paths only.
5. Validate before finishing: `python check_predictions_schema.py
   agent_reviews/<sid>.json`. If it prints PASS, you're done.
"""

_SCHEMA_VALIDATOR = '''\
#!/usr/bin/env python3
"""Schema validator for an ICLR agent review.

Usage: python check_predictions_schema.py agent_reviews/<sid>.json

Exits 0 on PASS, 1 on FAIL (printing what's wrong). The reviewer agent is
instructed to run this at the end of its review.
"""
import json, sys

REQUIRED_STR = ["decision", "summary", "strengths", "weaknesses",
                "questions", "missing_references"]
REQUIRED_INT = {
    "rating": (0, 10),
    "confidence": (1, 5),
    "soundness": (1, 4),
    "presentation": (1, 4),
    "contribution": (1, 4),
}


def main(path: str) -> int:
    try:
        with open(path) as f:
            obj = json.load(f)
    except Exception as e:
        print(f"FAIL: could not load {path}: {e}")
        return 1
    if not isinstance(obj, dict):
        print(f"FAIL: expected a JSON object, got {type(obj).__name__}")
        return 1
    errs = []
    for k in REQUIRED_STR:
        v = obj.get(k)
        if not isinstance(v, str) or not v.strip():
            errs.append(f"missing/empty string field: {k!r}")
    if obj.get("decision") not in ("accept", "reject"):
        errs.append(f"decision must be 'accept' or 'reject' (lowercase), got {obj.get('decision')!r}")
    for k, (lo, hi) in REQUIRED_INT.items():
        v = obj.get(k)
        if not isinstance(v, int) or isinstance(v, bool):
            errs.append(f"{k} must be an integer, got {type(v).__name__}={v!r}")
        elif not (lo <= v <= hi):
            errs.append(f"{k}={v} out of range [{lo},{hi}]")
    if errs:
        for e in errs:
            print(f"FAIL: {e}")
        return 1
    print(f"PASS: {path}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: check_predictions_schema.py <review.json>")
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
'''


# ---------------------------------------------------------------------------
# Workspace builder
# ---------------------------------------------------------------------------

def build_workspace(*, dest: Path, paperprep_output_dir: Path,
                    submission_id: str, modality: str,
                    panel_path: Optional[Path],
                    text_md_path: Optional[Path],
                    paper_title: Optional[str],
                    with_prior: bool,
                    prior_decision: Optional[str],
                    prior_p_accept: Optional[float],
                    prior_calibration: Optional[dict] = None,
                    domain: Optional[str] = None,
                    agent: str = "claude") -> dict:
    """Lay out one workspace on disk and return a manifest describing what's in it.

    Symlinks the page images and panel from the paperprep output dir to avoid
    re-encoding (ZHUANGL inode budget is tight; this keeps each workspace at
    ~handful of inodes regardless of page count).
    """
    dest.mkdir(parents=True, exist_ok=True)
    papers_dir = dest / "papers" / submission_id
    pages_dir = papers_dir / "page_images"
    papers_dir.mkdir(parents=True, exist_ok=True)
    pages_dir.mkdir(parents=True, exist_ok=True)
    (dest / "agent_reviews").mkdir(parents=True, exist_ok=True)

    # 1) text.md -- the normalized text paperprep produced for this paper.
    text_md_out = papers_dir / "text.md"
    if text_md_path and text_md_path.exists():
        try:
            text_md_out.write_text(text_md_path.read_text(encoding="utf-8", errors="replace"))
        except Exception as e:
            log.warning("workspace: could not copy text.md (%s)", e)
            text_md_out.write_text("")
    else:
        text_md_out.write_text("")

    # 2) Page images -- discover under paperprep's normalized/*/page_images/.
    src_pages = sorted(paperprep_output_dir.rglob("page_images/page_*.png"))
    image_rels: list[str] = []
    for src in src_pages:
        link = pages_dir / src.name
        try:
            if link.is_symlink() or link.exists():
                link.unlink()
            os.symlink(src, link)
        except OSError:
            try:
                link.write_bytes(src.read_bytes())
            except Exception as e:
                log.warning("workspace: could not include %s (%s)", src.name, e)
                continue
        image_rels.append(f"papers/{submission_id}/page_images/{src.name}")

    # 3) panel.png (optional)
    if panel_path and panel_path.exists():
        plink = dest / "panel.png"
        try:
            if plink.is_symlink() or plink.exists():
                plink.unlink()
            os.symlink(panel_path, plink)
        except OSError:
            try:
                plink.write_bytes(panel_path.read_bytes())
            except Exception:
                pass

    # 4) content.json -- the agent reads this first; mirrors cceval's shape.
    content: dict = {
        "submission_id": submission_id,
        "title": paper_title or "",
        "modality": modality,
        "content_text_path": f"papers/{submission_id}/text.md",
        "img_pages": image_rels,
    }
    if with_prior and prior_calibration:
        # Surface BOTH raw and Platt-calibrated p_accept. The agent's prompt
        # tells it which to trust (calibrated, since the paper's calibration
        # plots show it tracks accuracy roughly 1:1).
        cal = prior_calibration
        content["paperlens_prior"] = {
            "model": f"PaperLens-{modality[:1].upper()}",
            "domain": domain or "",
            "decision": cal.get("decision"),
            "p_accept_raw": cal.get("p_accept_raw"),
            "p_accept_cal": cal.get("p_accept_cal"),
            "confidence": cal.get("confidence"),
            "z": cal.get("z"),
            "platt_a": cal.get("platt_a"),
            "platt_b": cal.get("platt_b"),
            "has_calibration": cal.get("has_calibration", False),
        }
        # Legacy keys for backward-compat with the cceval scaffold's prompts.
        content["model_prediction"] = (cal.get("decision") or "accept").lower()
        content["model_confidence"] = cal.get("confidence")
    elif with_prior and prior_decision and prior_p_accept is not None:
        # No calibration bundle handed in (unusual, e.g. tests) -- fall back to
        # the raw values the caller had.
        content["model_prediction"] = prior_decision.lower()
        content["model_confidence"] = round(float(prior_p_accept), 4)
    (papers_dir / "content.json").write_text(json.dumps(content, indent=2))

    # 5) Schema validator + system rules + headless prompt.
    (dest / "check_predictions_schema.py").write_text(_SCHEMA_VALIDATOR)
    os.chmod(dest / "check_predictions_schema.py", 0o755)
    # Agents look for CLAUDE.md / AGENTS.md in cwd. We write BOTH so the user
    # can switch agents on the same workspace; the headless prompt itself is
    # passed via stdin/argv, not via these files. Title swaps "ICLR" for
    # "ML Conference" on arxiv-trained runs so the agent doesn't anchor on
    # ICLR-specific calibration the prompt no longer claims.
    title = "ICLR Paper Review -- Schema Reference" if (domain or "").lower() == "iclr" \
            else "ML Conference Paper Review -- Schema Reference"
    rules = _SYSTEM_RULES_BASE.format(title=title)
    (dest / "CLAUDE.md").write_text(rules)
    (dest / "AGENTS.md").write_text(rules)

    persona = _persona_for(domain)
    if with_prior:
        cal = prior_calibration or {}
        p_raw = cal.get("p_accept_raw")
        p_cal = cal.get("p_accept_cal")
        conf = cal.get("confidence")
        # When the (domain, modality) cell has no Platt row we leave the
        # calibrated line as "n/a" rather than silently lying.
        prompt = _PROMPT_WITH_PRIOR.format(
            sid=submission_id,
            decision=(cal.get("decision") or prior_decision or "accept").lower(),
            modality_letter=(modality[:1].upper() if modality else "V"),
            domain=(domain or "n/a"),
            p_raw=float(p_raw if p_raw is not None else (prior_p_accept or 0.5)),
            p_cal_disp=(f"{p_cal:.3f}" if p_cal is not None else "n/a (no Platt row for this cell)"),
            confidence_disp=(f"{conf:.3f}" if conf is not None else "n/a"),
            **persona,
        )
    else:
        prompt = _PROMPT_NO_PRIOR.format(sid=submission_id, **persona)

    return {
        "workspace": str(dest),
        "submission_id": submission_id,
        "with_prior": with_prior,
        "agent": agent,
        "prompt": prompt,
        "n_images": len(image_rels),
        "predictions_path": str(dest / "agent_reviews" / f"{submission_id}.json"),
    }


# ---------------------------------------------------------------------------
# Normalized event shape pushed to the UI
# ---------------------------------------------------------------------------

@dataclass
class AgentEvent:
    """One UI-renderable step in an agent's transcript."""
    seq: int
    ts: float
    kind: str           # "text" | "tool_call" | "tool_result" | "status" | "done" | "error"
    payload: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Stream parsers -- map raw stdout NDJSON to AgentEvents.
# ---------------------------------------------------------------------------

def _parse_claude_line(raw: dict) -> list[dict]:
    """Translate one Claude Code stream-json event to zero-or-more UI events.

    The interesting types are:
      - "system" subtype=init -> emit a "status" event with model + cwd
      - "assistant" -> walk message.content, splitting text vs tool_use blocks
      - "user" -> walk message.content for tool_result blocks
      - "result" -> emit "done" with the final text
    """
    out: list[dict] = []
    t = raw.get("type")
    if t == "system" and raw.get("subtype") == "init":
        out.append({"kind": "status",
                    "payload": {"model": raw.get("model"), "cwd": raw.get("cwd")}})
    elif t == "assistant":
        for block in (raw.get("message", {}).get("content") or []):
            bt = block.get("type")
            if bt == "text" and block.get("text"):
                out.append({"kind": "text", "payload": {"role": "assistant", "text": block["text"]}})
            elif bt == "tool_use":
                out.append({"kind": "tool_call",
                            "payload": {"id": block.get("id"), "name": block.get("name"),
                                        "input": block.get("input") or {}}})
    elif t == "user":
        for block in (raw.get("message", {}).get("content") or []):
            if block.get("type") == "tool_result":
                content = block.get("content")
                if isinstance(content, list):
                    parts = [b.get("text", "") for b in content if isinstance(b, dict)]
                    content = "\n".join(p for p in parts if p)
                out.append({"kind": "tool_result",
                            "payload": {"tool_use_id": block.get("tool_use_id"),
                                        "content": str(content or "")[:8000]}})
    elif t == "result":
        # Emit a "status" tombstone with cost/duration; the *real* terminal
        # "done" event (with predictions loaded from disk) is synthesized by
        # run_agent_headless after the process exits. This avoids the UI
        # rendering two final cards per run.
        out.append({"kind": "status",
                    "payload": {"msg": "claude finished",
                                "duration_ms": raw.get("duration_ms"),
                                "cost_usd": raw.get("total_cost_usd"),
                                "claude_result": raw.get("result") or ""}})
    return out


def _parse_codex_line(raw: dict) -> list[dict]:
    """Translate one ``codex exec --json`` event to zero-or-more UI events.

    Codex emits "item.started"/"item.completed" wrapping a typed item:
      - agent_message -> text
      - command_execution -> tool_call (on started) + tool_result (on completed)
      - file_change -> tool_call (on started) + tool_result (on completed)
    """
    out: list[dict] = []
    t = raw.get("type") or ""
    item = raw.get("item") or {}
    it = item.get("type")
    if t == "thread.started":
        out.append({"kind": "status", "payload": {"thread_id": raw.get("thread_id")}})
    elif t == "turn.started":
        pass
    elif t == "turn.completed":
        usage = raw.get("usage") or {}
        # Same as the claude parser: status, not done -- run_agent_headless
        # emits the single canonical done event after the process exits.
        out.append({"kind": "status",
                    "payload": {"msg": "codex turn complete",
                                "input_tokens": usage.get("input_tokens"),
                                "output_tokens": usage.get("output_tokens")}})
    elif t in ("item.started", "item.completed"):
        if it == "agent_message" and t == "item.completed":
            out.append({"kind": "text",
                        "payload": {"role": "assistant", "text": item.get("text") or ""}})
        elif it == "command_execution":
            if t == "item.started":
                out.append({"kind": "tool_call",
                            "payload": {"id": item.get("id"), "name": "Bash",
                                        "input": {"command": item.get("command")}}})
            else:
                out.append({"kind": "tool_result",
                            "payload": {"tool_use_id": item.get("id"),
                                        "content": (item.get("aggregated_output") or "")[:8000],
                                        "exit_code": item.get("exit_code")}})
        elif it == "file_change":
            if t == "item.started":
                out.append({"kind": "tool_call",
                            "payload": {"id": item.get("id"), "name": "FileChange",
                                        "input": {"changes": item.get("changes")}}})
            else:
                out.append({"kind": "tool_result",
                            "payload": {"tool_use_id": item.get("id"),
                                        "content": "applied"}})
    return out


# ---------------------------------------------------------------------------
# Runner: spawn the headless CLI, parse stdout, push events.
# ---------------------------------------------------------------------------

@dataclass
class AgentResult:
    ok: bool
    exit_code: int
    final_text: str
    error: Optional[str] = None
    predictions: Optional[dict] = None    # parsed agent_reviews/<sid>.json
    workspace: Optional[str] = None


def _build_argv(agent: str, prompt: str, workspace: Path) -> list[str]:
    """Headless invocation with matched xhigh reasoning effort on both
    CLIs. The paper used "high reasoning effort" for both Claude Code
    (Sonnet-4.6) and Codex (GPT-5.4); we go one step above that on each.
    """
    if agent == "claude":
        return [
            "claude", "-p", prompt,
            "--output-format", "stream-json",
            "--verbose",
            "--effort", "xhigh",              # low|medium|high|xhigh|max
            "--dangerously-skip-permissions",
            "--add-dir", str(workspace),
        ]
    if agent == "codex":
        return [
            "codex", "exec", "--json",
            "--skip-git-repo-check",
            "--dangerously-bypass-approvals-and-sandbox",
            "-c", "model_reasoning_effort=xhigh",   # GPT-5.x: xhigh tier
            "-C", str(workspace),
            prompt,
        ]
    raise ValueError(f"unknown agent {agent!r}")


def run_agent_headless(
    *, agent: str, workspace: Path, prompt: str, submission_id: str,
    on_event: Callable[[dict], None],
    timeout_s: float = 1800.0,
) -> AgentResult:
    """Spawn the agent CLI in headless mode, stream stdout, return its verdict.

    Each parsed event is pushed through ``on_event`` immediately (the server's
    ring buffer wraps that callback). Stdout is consumed in this thread, so
    callers should park this function in its own thread per workspace.
    """
    argv = _build_argv(agent, prompt, workspace)
    parser = _parse_claude_line if agent == "claude" else _parse_codex_line
    started = time.time()
    on_event({"kind": "status",
              "payload": {"msg": f"launching {agent}", "argv0": argv[0]}})
    env = dict(os.environ)
    env.setdefault("CI", "1")
    final_text = ""
    error: Optional[str] = None
    proc: Optional[subprocess.Popen] = None
    try:
        proc = subprocess.Popen(argv, cwd=str(workspace),
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, bufsize=1, env=env)
    except FileNotFoundError as e:
        on_event({"kind": "error", "payload": {"msg": f"agent binary not found: {e}"}})
        return AgentResult(ok=False, exit_code=127, final_text="",
                           error=f"binary not found: {e}", workspace=str(workspace))

    assert proc.stdout is not None
    stderr_buf: list[str] = []
    if proc.stderr is not None:
        def _drain_stderr() -> None:
            try:
                for line in proc.stderr:  # type: ignore[union-attr]
                    stderr_buf.append(line)
                    if len(stderr_buf) > 200:
                        del stderr_buf[:100]
            except Exception:
                pass
        threading.Thread(target=_drain_stderr, daemon=True,
                         name=f"agent-stderr-{agent}").start()

    deadline = started + timeout_s
    try:
        for line in proc.stdout:
            if time.time() > deadline:
                proc.kill()
                error = "agent timed out"
                on_event({"kind": "error", "payload": {"msg": error}})
                break
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            for ev in parser(raw):
                # Track the last assistant text -- becomes final_text on the
                # synthesized done event below. Tool calls/results don't
                # count; the "Done: <sid>" line is what we want.
                if ev.get("kind") == "text":
                    txt = (ev.get("payload") or {}).get("text") or ""
                    if txt.strip():
                        final_text = txt.strip().splitlines()[-1][:240]
                on_event(ev)
    except Exception as e:
        error = f"stream read failed: {e}"
        log.exception("agent stream read failed")
        on_event({"kind": "error", "payload": {"msg": error}})

    rc = proc.wait()
    if rc != 0 and not error:
        tail = "".join(stderr_buf)[-1000:]
        error = f"agent exited {rc}; stderr tail: {tail}"
        on_event({"kind": "error", "payload": {"msg": error}})

    predictions = _read_predictions(workspace, submission_id)
    on_event({"kind": "done", "payload": {"final_text": final_text,
                                          "predictions": predictions,
                                          "elapsed_s": round(time.time() - started, 2)}})
    return AgentResult(ok=(rc == 0 and predictions is not None),
                       exit_code=rc, final_text=final_text, error=error,
                       predictions=predictions, workspace=str(workspace))


def _read_predictions(workspace: Path, submission_id: str) -> Optional[dict]:
    """Best-effort load of the agent's written review JSON. Returns None if the
    agent didn't produce one (we surface that as a failed agent run upstream).
    """
    candidates = [
        workspace / "agent_reviews" / f"{submission_id}.json",
        workspace / "PREDICTIONS.json",
    ]
    for p in candidates:
        if p.exists():
            try:
                return json.loads(p.read_text())
            except Exception as e:
                log.warning("predictions parse failed (%s): %s", p, e)
    return None
