"""Reranker is now an HTTP client to ``paperlens serve``. Tests use a mock
/score endpoint so they run without a GPU.

For an end-to-end test against a real running serve, see
``paperlens-training-and-inference/scripts/tests/test_serve_idempotency.py``
which gates on the ``PAPERLENS_SERVE_URL`` env var.
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional

import pytest

from paperlens_arxiv_server.reranker import PaperLensReranker
from paperlens_arxiv_server.retriever_client import RetrievedPaper


# ---------------------------------------------------------------------------
# Mock /score server -- pure-stdlib, no GPU, no vLLM
# ---------------------------------------------------------------------------

class _MockScoreHandler(BaseHTTPRequestHandler):
    """Returns one scripted PaperScore per request paper.

    Capture mode (server attaches `received` list): every POST request's
    body is appended so tests can assert what the reranker sent.
    """

    server_version = "MockScore/1.0"

    def log_message(self, *args, **kwargs):  # silence test output
        pass

    def do_GET(self):
        if self.path == "/health":
            body = {
                "status": "ok",
                "ckpt_path": "/mock/ckpt-1",
                "template": "qwen2_vl",
                "decision_token_idx": 5,
                "compute_arch": "mock_test",
            }
            self._json(200, body)
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/score":
            self._json(404, {"error": "not found"})
            return
        length = int(self.headers.get("content-length", 0))
        body = json.loads(self.rfile.read(length))
        self.server.received.append(body)
        # Scripted scores: pre-loaded onto the server before this handler runs.
        n = len(body["papers"])
        scores = self.server.scripted_scores[:n]
        self._json(200, {"scores": scores})

    def _json(self, code: int, body: dict):
        payload = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.fixture
def mock_serve():
    httpd = HTTPServer(("127.0.0.1", 0), _MockScoreHandler)
    httpd.scripted_scores = []
    httpd.received = []
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        yield (httpd, f"http://127.0.0.1:{port}")
    finally:
        httpd.shutdown()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_reranker_health_probe(mock_serve):
    _, url = mock_serve
    rr = PaperLensReranker(serve_url=url, domain="arxiv", modality="vision")
    assert rr.compute_arch == "mock_test"


def test_score_returns_floats(mock_serve):
    httpd, url = mock_serve
    httpd.scripted_scores = [
        {"p_accept": 0.82, "logp_accept": -0.5, "logp_reject": -2.1, "pred": "Outcome: \\boxed{Accept}"},
        {"p_accept": 0.18, "logp_accept": -3.0, "logp_reject": -0.5, "pred": "Outcome: \\boxed{Reject}"},
    ]
    rr = PaperLensReranker(serve_url=url, domain="arxiv", modality="vision")
    papers = [
        RetrievedPaper(paper_id="2305.00001", title="A", abstract="ab", score=0.9, images=["/x/page_1.png"]),
        RetrievedPaper(paper_id="2305.00002", title="B", abstract="bc", score=0.8, images=["/x/page_1.png"]),
    ]
    scores = rr.score(papers)
    assert scores == [0.82, 0.18]


def test_sharegpt_row_has_required_fields(mock_serve):
    """The body the reranker POSTs must look like a paperprep-style sharegpt row:
    system + human + gpt turns, _metadata with arxiv_id, images list."""
    httpd, url = mock_serve
    httpd.scripted_scores = [{"p_accept": 0.5}]
    rr = PaperLensReranker(serve_url=url, domain="arxiv", modality="vision")
    rr.score([RetrievedPaper(
        paper_id="2305.00007", title="T", abstract="A", score=0.5,
        images=["/x/page_1.png", "/x/page_2.png"],
    )])
    body = httpd.received[-1]
    assert "papers" in body and len(body["papers"]) == 1
    row = body["papers"][0]
    assert "conversations" in row and len(row["conversations"]) >= 2
    roles = [c["from"] for c in row["conversations"]]
    assert "system" in roles and "human" in roles
    # vision rows must carry image paths
    assert row.get("images") == ["/x/page_1.png", "/x/page_2.png"]
    assert row["_metadata"]["arxiv_id"] == "2305.00007"
    # human turn should contain the ARXIV prompt + the abstract + N <image>
    human = next(c["value"] for c in row["conversations"] if c["from"] == "human")
    assert "acceptance outcome" in human
    assert "## Abstract\nA" in human
    assert human.count("<image>") == 2


def test_iclr_prompt_used_when_domain_is_iclr(mock_serve):
    httpd, url = mock_serve
    httpd.scripted_scores = [{"p_accept": 0.5}]
    rr = PaperLensReranker(serve_url=url, domain="iclr", modality="vision")
    rr.score([RetrievedPaper(
        paper_id="xyz", title="T", abstract="A", score=0.5, images=["/x/page_1.png"],
    )])
    human = next(c["value"] for c in httpd.received[-1]["papers"][0]["conversations"] if c["from"] == "human")
    assert "acceptance outcome at ICLR" in human


def test_empty_papers_returns_empty(mock_serve):
    _, url = mock_serve
    rr = PaperLensReranker(serve_url=url, domain="arxiv", modality="vision")
    assert rr.score([]) == []
