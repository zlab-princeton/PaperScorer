"""Smoke test for `paperprep serve` — spawn the server, hit /prepare, shut down.

Tagged `gpu`. Wall time: ~5-8 min on an L40 (cold MinerU warmup + two
inference requests; the 3-paper request includes one latex_dir compile).
Full per-stage output correctness is already covered by
``test_golden_e2e_pdf.py`` and ``test_golden_e2e_latex.py`` — this file
only verifies the HTTP plumbing across all three input types.

  uv run pytest tests/integration/test_serve_smoke.py -m gpu --no-header -s
"""
from __future__ import annotations

import gzip
import json
import os
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from urllib import error as urlerror
from urllib import request as urlrequest

import pytest

from tests.integration.conftest import (
    ARXIV_IDS,
    ICLR_IDS,
    INPUTS_DIR,
)


pytestmark = pytest.mark.gpu


DEFAULT_TEXLIVE_BIN = os.environ.get("PAPERPREP_TEXLIVE_BIN", "")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _http_get(url: str, timeout: float = 5.0) -> tuple[int, dict | None]:
    try:
        with urlrequest.urlopen(url, timeout=timeout) as r:
            body = json.loads(r.read().decode())
            return r.status, body
    except urlerror.HTTPError as e:
        return e.code, None
    except (urlerror.URLError, ConnectionError, OSError):
        return 0, None


def _http_post(url: str, payload: dict, timeout: float = 600.0) -> tuple[int, dict | None]:
    data = json.dumps(payload).encode()
    req = urlrequest.Request(url, data=data, method="POST",
                              headers={"Content-Type": "application/json"})
    try:
        with urlrequest.urlopen(req, timeout=timeout) as r:
            body = json.loads(r.read().decode())
            return r.status, body
    except urlerror.HTTPError as e:
        try:
            body = json.loads(e.read().decode())
        except Exception:
            body = None
        return e.code, body


def _wait_healthy(port: int, timeout: float = 600.0) -> None:
    """Block until /healthz returns 200 (server + MinerU both ready)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        status, _ = _http_get(f"http://127.0.0.1:{port}/healthz", timeout=2)
        if status == 200:
            return
        time.sleep(2)
    pytest.fail(f"serve /healthz did not return 200 within {timeout}s")


# ---------------------------------------------------------------------------
# fixture: launch one server, share across the two tests in this module
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def serve_subprocess(mineru_available):
    """Spawn `paperprep serve` and yield its port. Logs to an NFS path
    (not the per-job /tmp) so failures survive SLURM OOM / job teardown.
    Set ``PAPERPREP_SERVE_LOG_DIR`` to override the default location.
    """
    # Default to a per-test tmp dir; override on a real cluster via
    # PAPERPREP_SERVE_LOG_DIR so failures survive SLURM OOM / job teardown.
    out_dir = Path(os.environ.get(
        "PAPERPREP_SERVE_LOG_DIR",
        tempfile.mkdtemp(prefix="paperprep_serve_smoke_"),
    ))
    out_dir.mkdir(parents=True, exist_ok=True)
    flask_port = _free_port()
    mineru_port = _free_port()
    texlive_bin = os.environ.get("PAPERPREP_TEXLIVE_BIN", DEFAULT_TEXLIVE_BIN)
    serve_stdout = out_dir / "serve.stdout.log"

    cmd = [
        "uv", "run", "--extra", "gpu", "--extra", "serve", "--no-sync",
        "paperprep", "serve",
        "--output-dir", str(out_dir),
        "--port", str(flask_port),
        "--mineru-port", str(mineru_port),
        "--gpu-memory-utilization", "0.6",
        "--texlive-bin", texlive_bin,
    ]
    repo_root = Path(__file__).resolve().parent.parent.parent
    log = open(serve_stdout, "w")
    print(f"\n[serve_subprocess] cmd: {' '.join(cmd)}")
    print(f"[serve_subprocess] log: {serve_stdout}")
    proc = subprocess.Popen(
        cmd, cwd=str(repo_root), stdout=log, stderr=subprocess.STDOUT,
    )
    try:
        try:
            _wait_healthy(flask_port, timeout=600)
        except Exception:
            # Surface the server log so the user can see why it didn't come up
            print(f"\n===== serve.stdout.log tail ({serve_stdout}) =====")
            try:
                tail = serve_stdout.read_text().splitlines()[-80:]
                for line in tail:
                    print(line)
            except Exception as e:
                print(f"(could not read log: {e})")
            raise
        yield {"port": flask_port, "mineru_port": mineru_port, "output_dir": out_dir}
    finally:
        # Best-effort graceful shutdown
        try:
            _http_post(f"http://127.0.0.1:{flask_port}/shutdown", {}, timeout=10)
        except Exception:
            pass
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
        log.close()


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def test_healthz_returns_ok(serve_subprocess):
    status, body = _http_get(f"http://127.0.0.1:{serve_subprocess['port']}/healthz")
    assert status == 200
    assert body is not None
    assert body["status"] == "ok"
    assert "mineru_url" in body
    assert body["uptime_s"] >= 0


def test_prepare_single_pdf(serve_subprocess):
    aid = ARXIV_IDS[0]
    pdf = INPUTS_DIR / f"{aid}.pdf"
    assert pdf.exists(), f"fixture PDF missing: {pdf}"

    status, body = _http_post(
        f"http://127.0.0.1:{serve_subprocess['port']}/prepare",
        {"id": aid, "type": "pdf", "path": str(pdf)},
        timeout=300,
    )
    assert status == 200, f"got {status}; body={body}"
    assert body is not None
    assert body["sharegpt_text_path"], f"missing sharegpt_text_path; body={body}"
    assert Path(body["sharegpt_text_path"]).exists()
    assert Path(body["sharegpt_vision_path"]).exists()
    assert len(body["papers"]) == 1
    assert body["papers"][0]["status"] == "ok", f"paper failed: {body['papers'][0]}"
    assert Path(body["papers"][0]["normalized_md"]).exists()


def _extract_latex_src(aid: str, dest_root: Path) -> Path:
    """Extract a fixture .gz to a fresh dir; mirrors test_golden_e2e_latex."""
    gz_path = INPUTS_DIR / f"{aid}.gz"
    dest = dest_root / aid
    dest.mkdir()
    try:
        with tarfile.open(str(gz_path)) as tf:
            tf.extractall(str(dest), filter="data")
    except tarfile.ReadError:
        with gzip.open(str(gz_path), "rb") as gz:
            (dest / f"{aid}.tex").write_bytes(gz.read())
    return dest


def test_prepare_manifest_iclr_pdf_arxiv_pdf_arxiv_latex(serve_subprocess, tmp_path):
    """One ICLR PDF, one arXiv PDF, one arXiv LaTeX source dir — exercises
    all three input-type codepaths in a single request."""
    iclr_id = ICLR_IDS[0]
    arxiv_pdf_id = ARXIV_IDS[0]
    arxiv_latex_id = ARXIV_IDS[1]

    latex_src_root = tmp_path / "latex_sources"
    latex_src_root.mkdir()
    latex_dir = _extract_latex_src(arxiv_latex_id, latex_src_root)

    payload = {"papers": [
        {"id": iclr_id, "type": "pdf",
         "path": str(INPUTS_DIR / f"{iclr_id}.pdf")},
        {"id": f"{arxiv_pdf_id}_pdf", "type": "pdf",
         "path": str(INPUTS_DIR / f"{arxiv_pdf_id}.pdf")},
        {"id": f"{arxiv_latex_id}_latex", "type": "latex_dir",
         "path": str(latex_dir)},
    ]}
    status, body = _http_post(
        f"http://127.0.0.1:{serve_subprocess['port']}/prepare",
        payload, timeout=900,
    )
    assert status == 200, f"got {status}; body={body}"
    assert body is not None
    assert len(body["papers"]) == 3
    statuses = [p["status"] for p in body["papers"]]
    assert statuses == ["ok", "ok", "ok"], f"per-paper: {body['papers']}"
    # text/vision JSONs should contain all 3 paper ids
    text_data = json.loads(Path(body["sharegpt_text_path"]).read_text())
    ids = {e["_metadata"]["id"] for e in text_data}
    expected = {iclr_id, f"{arxiv_pdf_id}_pdf", f"{arxiv_latex_id}_latex"}
    assert ids == expected, f"missing ids: {expected - ids}"
