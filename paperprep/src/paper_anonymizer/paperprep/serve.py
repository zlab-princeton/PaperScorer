"""`paperprep serve` — persistent HTTP wrapper around the prep pipeline.

Spawns ``mineru-vllm-server`` as a child process so the VLM weights load
exactly once at startup, then exposes a Flask app with:

  GET  /healthz   -> {"status": "ok", "mineru_url": ..., "uptime_s": ...}
  POST /prepare   -> runs compile/mineru/normalize/filter/export for one
                     paper or a small manifest; returns paths to outputs
  POST /shutdown  -> tears the server (and MinerU subprocess) down cleanly

The existing batch CLI (`paperprep run`) is untouched. The two share the
same per-paper stage functions; serve mode bypasses StateLog + submitit
and runs each request synchronously on the Flask thread.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest

from paper_anonymizer.core.latex_compile import get_pdf_page_count
from paper_anonymizer.paperprep import manifest as mfst
from paper_anonymizer.paperprep.stages import (
    compile as stage_compile,
    export as stage_export,
    filter as stage_filter,
    mineru as stage_mineru,
    normalize as stage_normalize,
)


_MIN_PAGES = 4   # page-count gate matches stages/detect.MIN_PAGES


# ---------------------------------------------------------------------------
# Process-wide state (Flask + signal handlers both read this)
# ---------------------------------------------------------------------------

class _ServerState:
    def __init__(self) -> None:
        self.started_at: float = time.time()
        self.mineru_proc: subprocess.Popen | None = None
        self.shutting_down: bool = False
        self.lock = threading.Lock()


_state = _ServerState()


# ---------------------------------------------------------------------------
# GPU + MinerU lifecycle
# ---------------------------------------------------------------------------

def _assert_gpu() -> None:
    """Fail fast if no NVIDIA GPU is visible.

    Uses ``nvidia-smi -L`` to avoid pulling in torch just for the check.
    """
    try:
        r = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=5,
        )
    except FileNotFoundError:
        print("ERROR: nvidia-smi not on PATH — paperprep serve requires a GPU node.",
              file=sys.stderr)
        sys.exit(2)
    except subprocess.TimeoutExpired:
        print("ERROR: nvidia-smi timed out.", file=sys.stderr)
        sys.exit(2)
    if r.returncode != 0 or not r.stdout.strip():
        print(f"ERROR: no GPU detected by nvidia-smi (rc={r.returncode}).\n"
              f"stdout: {r.stdout}\nstderr: {r.stderr}", file=sys.stderr)
        sys.exit(2)
    print(f"[paperprep serve] GPU(s): {r.stdout.strip()}")


def _start_mineru_server(port: int, gpu_mem: float, logs_dir: Path) -> subprocess.Popen:
    """Spawn ``mineru-vllm-server`` in its own session so it survives child
    teardown nuance and can be killed as a group on shutdown.
    """
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / f"mineru-vllm-server.{int(time.time())}.log"
    print(f"[paperprep serve] launching mineru-vllm-server on port {port} "
          f"(gpu-memory-utilization={gpu_mem}); log: {log_file}")
    f = open(log_file, "w", buffering=1)
    proc = subprocess.Popen(
        [
            "mineru-vllm-server",
            "--port", str(port),
            "--gpu-memory-utilization", str(gpu_mem),
        ],
        stdout=f, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    # Tail the log into our stdout in a background thread so the operator
    # sees the vLLM warmup without `tail -f`-ing manually.
    threading.Thread(target=_tail_log, args=(log_file, proc), daemon=True).start()
    return proc


def _tail_log(log_path: Path, proc: subprocess.Popen) -> None:
    """Stream the MinerU server log to stdout, prefixed `[mineru]`."""
    # Wait for the file to exist (the Popen above just created it but be safe)
    while not log_path.exists() and proc.poll() is None:
        time.sleep(0.2)
    try:
        with open(log_path) as f:
            while proc.poll() is None and not _state.shutting_down:
                line = f.readline()
                if not line:
                    time.sleep(0.5)
                    continue
                print(f"[mineru] {line.rstrip()}")
    except Exception as e:
        print(f"[paperprep serve] log tailer exited: {e}")


def _wait_mineru_ready(port: int, timeout: int = 600) -> None:
    """Poll ``/health`` then ``/v1/models`` until the server responds 200.

    vLLM's OpenAI-compatible server exposes ``/health`` and ``/v1/models``
    once the model is fully loaded into GPU memory (~30-60 s for Qwen2-VL).
    """
    deadline = time.time() + timeout
    last_msg = 0.0
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        for path in ("/health", "/v1/models"):
            url = f"http://127.0.0.1:{port}{path}"
            try:
                with urlrequest.urlopen(url, timeout=2) as r:
                    if 200 <= r.status < 300:
                        elapsed = int(time.time() - _state.started_at)
                        print(f"[paperprep serve] MinerU ready after {elapsed}s "
                              f"(probe={path})")
                        return
            except (urlerror.URLError, ConnectionError, TimeoutError, OSError):
                pass
        # Make sure the subprocess hasn't died on us
        if _state.mineru_proc is not None and _state.mineru_proc.poll() is not None:
            print(f"ERROR: mineru-vllm-server exited with code "
                  f"{_state.mineru_proc.returncode} during warmup.", file=sys.stderr)
            sys.exit(3)
        now = time.time()
        if now - last_msg >= 10:
            elapsed = int(now - _state.started_at)
            print(f"[paperprep serve] MinerU warming up… {elapsed}s elapsed "
                  f"(attempt {attempt})")
            last_msg = now
        time.sleep(2)
    print(f"ERROR: mineru-vllm-server did not become ready within {timeout}s.",
          file=sys.stderr)
    sys.exit(3)


def _shutdown_mineru(proc: subprocess.Popen | None) -> None:
    """Send SIGTERM to the MinerU process group; escalate to SIGKILL if slow."""
    if proc is None or proc.poll() is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    print(f"[paperprep serve] shutting down mineru-vllm-server (pgid={pgid})")
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        print("[paperprep serve] mineru-vllm-server didn't exit on SIGTERM; sending SIGKILL")
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            return
        proc.wait(timeout=5)


def _install_signal_handlers() -> None:
    def _handler(signum, _frame):
        if _state.shutting_down:
            return
        _state.shutting_down = True
        print(f"\n[paperprep serve] received signal {signum}; shutting down")
        _shutdown_mineru(_state.mineru_proc)
        # Re-raise the default behavior so Flask's dev server exits
        sys.exit(0)
    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


# ---------------------------------------------------------------------------
# Per-request pipeline driver
# ---------------------------------------------------------------------------

def _coerce_papers(body: dict) -> list[dict]:
    """Accept either {papers: [...]} or a single-paper dict."""
    if isinstance(body.get("papers"), list):
        return body["papers"]
    return [body]


def _run_one_paper(
    p: dict,
    out_dir: Path,
    mineru_url: str,
    texlive_bin: str | None,
    mineru_max_pages: int,
    min_body_pages: int,
) -> tuple[dict, dict | None]:
    """Process one paper through compile/mineru/normalize/filter.

    Returns (response_dict_for_this_paper, export_item_or_None). The export
    item is None on failure; the caller batches all successful items into
    a single ShareGPT export call at the end.
    """
    paper_id = p.get("id") or f"req_{uuid.uuid4().hex[:8]}"
    try:
        entry = mfst.PaperEntry(
            id=paper_id,
            type=p["type"],
            path=Path(p["path"]),
            main_tex=p.get("main_tex"),
            extra=p.get("extra") or {},
        )
    except KeyError as e:
        return {"id": paper_id, "status": "failed",
                "error": f"missing required field: {e}"}, None

    if not entry.path.exists():
        return {"id": paper_id, "status": "failed",
                "error": f"path does not exist: {entry.path}"}, None

    # ---- compile -----------------------------------------------------------
    try:
        sr = stage_compile._compile_one(
            entry, out_dir, timeout=600, texlive_bin=texlive_bin,
        )
    except Exception as e:
        return {"id": paper_id, "status": "failed",
                "error": f"compile raised: {e}"}, None
    if sr.status == "failed":
        return {"id": paper_id, "status": "failed",
                "error": f"compile: {sr.error}"}, None
    # For type=pdf, _compile_one returns status="skipped" — passthrough.
    if sr.status == "ok" and sr.output_path:
        pdf_path = Path(sr.output_path)
    else:
        pdf_path = entry.path

    # ---- page-count gate (inline, matches stages/detect) -------------------
    pc = get_pdf_page_count(pdf_path)
    if pc is None:
        return {"id": paper_id, "status": "failed",
                "error": "pdfinfo could not read page count"}, None
    if pc < _MIN_PAGES:
        return {"id": paper_id, "status": "failed",
                "error": f"page_count={pc} < {_MIN_PAGES}"}, None

    # ---- mineru via persistent HTTP server ---------------------------------
    # Per-request page-cap override: a paper may carry its own ``max_pages``
    # (e.g. to score the first 16 pages instead of the server default 14).
    # Falls back to the server-level ``mineru_max_pages`` when unset/invalid.
    eff_max_pages = mineru_max_pages
    if p.get("max_pages") is not None:
        try:
            eff_max_pages = max(1, int(p["max_pages"]))
        except (TypeError, ValueError):
            eff_max_pages = mineru_max_pages
    try:
        sr = stage_mineru._mineru_one(
            paper_id, pdf_path, out_dir,
            max_pages=eff_max_pages,
            mineru_server_url=mineru_url,
        )
    except Exception as e:
        return {"id": paper_id, "status": "failed",
                "error": f"mineru raised: {e}"}, None
    if sr.status != "ok":
        return {"id": paper_id, "status": "failed",
                "error": f"mineru: {sr.error}"}, None
    mineru_cl = Path(sr.extra["content_list_path"])
    middle = Path(sr.extra["middle_json_path"]) if sr.extra.get("middle_json_path") else None

    # ---- normalize ---------------------------------------------------------
    try:
        sr = stage_normalize._normalize_one(
            paper_id, pdf_path, mineru_cl, middle, out_dir, dpi=150,
        )
    except Exception as e:
        return {"id": paper_id, "status": "failed",
                "error": f"normalize raised: {e}"}, None
    if sr.status != "ok":
        return {"id": paper_id, "status": "failed",
                "error": f"normalize: {sr.error}"}, None

    # ---- filter (uses the *normalized* content_list, matches cli.py) -------
    normalized_cl = out_dir / "normalized" / paper_id / f"{paper_id}_content_list.json"
    images_dir = out_dir / "normalized" / paper_id / "page_images"
    try:
        sr = stage_filter._filter_one(
            paper_id, normalized_cl, images_dir, min_body_pages=min_body_pages,
        )
    except Exception as e:
        return {"id": paper_id, "status": "failed",
                "error": f"filter raised: {e}"}, None
    if sr.status != "ok":
        return {"id": paper_id, "status": "failed",
                "error": f"filter: {sr.error}",
                "body_pages": (sr.extra or {}).get("body_pages")}, None

    body_pages = sr.extra["body_pages"]
    return (
        {
            "id": paper_id,
            "status": "ok",
            "normalized_md": str(out_dir / "normalized" / paper_id / f"{paper_id}.md"),
            "redacted_pdf": str(out_dir / "normalized" / paper_id / f"{paper_id}_redacted.pdf"),
            "body_pages": body_pages,
            "max_pages": eff_max_pages,
        },
        {
            "id": paper_id,
            "content_list_path": normalized_cl,
            "images_dir": images_dir,
            "body_pages": body_pages,
            "extra": entry.extra,
        },
    )


def _run_one_request(
    body: dict,
    base_output_dir: Path,
    mineru_url: str,
    texlive_bin: str | None,
    mineru_max_pages: int,
    min_body_pages: int,
) -> dict:
    request_id = body.get("request_id") or (
        time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    )
    out_dir = base_output_dir / request_id
    out_dir.mkdir(parents=True, exist_ok=True)

    papers = _coerce_papers(body)
    started = time.time()
    paper_results: list[dict] = []
    export_items: list[dict] = []
    for p in papers:
        pr, item = _run_one_paper(
            p, out_dir, mineru_url, texlive_bin, mineru_max_pages, min_body_pages,
        )
        paper_results.append(pr)
        if item is not None:
            export_items.append(item)

    text_path: Path | None = None
    vision_path: Path | None = None
    if export_items:
        try:
            text_entries, vision_entries = stage_export.build_entries(export_items)
            text_path, vision_path = stage_export.write_outputs(
                text_entries, vision_entries, out_dir,
            )
        except Exception as e:
            return {
                "request_id": request_id,
                "output_dir": str(out_dir),
                "error": f"export raised: {e}",
                "papers": paper_results,
                "elapsed_s": round(time.time() - started, 2),
            }

    return {
        "request_id": request_id,
        "output_dir": str(out_dir),
        "sharegpt_text_path": str(text_path) if text_path else None,
        "sharegpt_vision_path": str(vision_path) if vision_path else None,
        "papers": paper_results,
        "elapsed_s": round(time.time() - started, 2),
    }


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

def _make_app(
    output_dir: Path,
    mineru_url: str,
    texlive_bin: str | None,
    mineru_max_pages: int,
    min_body_pages: int,
) -> "Any":
    try:
        from flask import Flask, jsonify, request
    except ImportError as e:
        print("ERROR: flask not installed. Run `uv sync --extra serve`.",
              file=sys.stderr)
        raise SystemExit(2) from e

    app = Flask("paperprep-serve")

    @app.get("/healthz")
    def healthz():
        return jsonify({
            "status": "shutting_down" if _state.shutting_down else "ok",
            "mineru_url": mineru_url,
            "uptime_s": int(time.time() - _state.started_at),
        })

    @app.post("/prepare")
    def prepare():
        if _state.shutting_down:
            return jsonify({"error": "server is shutting down"}), 503
        body = request.get_json(force=True, silent=True)
        if not isinstance(body, dict):
            return jsonify({"error": "request body must be a JSON object"}), 400
        try:
            result = _run_one_request(
                body, output_dir, mineru_url, texlive_bin,
                mineru_max_pages, min_body_pages,
            )
        except Exception as e:
            return jsonify({"error": f"request handler raised: {e}"}), 500
        return jsonify(result)

    @app.post("/shutdown")
    def shutdown():
        if _state.shutting_down:
            return jsonify({"status": "already_shutting_down"})
        _state.shutting_down = True
        # Kill the MinerU subprocess synchronously so the operator gets a
        # clean response, then schedule Flask exit in a thread.
        _shutdown_mineru(_state.mineru_proc)
        threading.Thread(
            target=lambda: (time.sleep(0.5), os.kill(os.getpid(), signal.SIGTERM)),
            daemon=True,
        ).start()
        return jsonify({"status": "shutting_down"})

    return app


# ---------------------------------------------------------------------------
# Entry point (called by cli.py)
# ---------------------------------------------------------------------------

def run_serve(
    output_dir: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 5000,
    mineru_port: int = 30000,
    mineru_max_pages: int = 14,
    min_body_pages: int = 6,
    texlive_bin: str | None = None,
    gpu_memory_utilization: float = 0.7,
) -> int:
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"

    _assert_gpu()
    _install_signal_handlers()

    _state.mineru_proc = _start_mineru_server(mineru_port, gpu_memory_utilization, logs_dir)
    _wait_mineru_ready(mineru_port)

    mineru_url = f"http://127.0.0.1:{mineru_port}"
    app = _make_app(output_dir, mineru_url, texlive_bin, mineru_max_pages, min_body_pages)
    print(f"[paperprep serve] Flask listening on http://{host}:{port}  "
          f"(output_dir={output_dir})")
    try:
        # threaded=True so concurrent /prepare requests share the persistent
        # MinerU server — vLLM batches under the hood.
        app.run(host=host, port=port, threaded=True, use_reloader=False)
    finally:
        if not _state.shutting_down:
            _state.shutting_down = True
            _shutdown_mineru(_state.mineru_proc)
    return 0
