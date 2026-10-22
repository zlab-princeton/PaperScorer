"""End-to-end test against a running paperlens-arxiv-server.

Skipped unless ``PAPERLENS_E2E_URL`` is set (e.g. ``http://localhost:8000``).
Requires arxiv_retriever to also be reachable per the running server's config.
"""
from __future__ import annotations

import os
import pytest
import requests

pytestmark = pytest.mark.skipif(
    not os.environ.get("PAPERLENS_E2E_URL"),
    reason="set PAPERLENS_E2E_URL (e.g. http://localhost:8000) to run",
)


def test_health():
    url = os.environ["PAPERLENS_E2E_URL"].rstrip("/")
    r = requests.get(f"{url}/health", timeout=10)
    r.raise_for_status()
    body = r.json()
    assert body["status"] == "ok"
    for k in ("reranker_ckpt", "ckpt_id", "retriever_url", "images_root", "cache_rows_for_ckpt"):
        assert k in body, f"missing {k} in /health response"


def test_search_returns_blended_results():
    url = os.environ["PAPERLENS_E2E_URL"].rstrip("/")
    payload = {
        "query": "sparse attention transformer",
        "k": 10,
    }
    r = requests.post(f"{url}/search", json=payload, timeout=600)
    r.raise_for_status()
    body = r.json()

    assert body["n_retrieved"] > 0
    assert body["n_returned"] == min(10, body["n_retrieved"])
    assert len(body["results"]) == body["n_returned"]
    assert body["n_cache_hits"] + body["n_inferred"] == body["n_retrieved"]

    # Sort invariant: results sorted by blend_score desc
    blends = [r["blend_score"] for r in body["results"]]
    assert blends == sorted(blends, reverse=True), "results not sorted by blend_score"

    # p_accept in [0,1]
    for p in body["results"]:
        assert 0.0 <= p["p_accept"] <= 1.0


def test_repeat_query_hits_cache():
    """Second identical call should have n_inferred=0 (everything cached)."""
    url = os.environ["PAPERLENS_E2E_URL"].rstrip("/")
    payload = {"query": "attention is all you need", "k": 5}
    requests.post(f"{url}/search", json=payload, timeout=600).raise_for_status()
    r = requests.post(f"{url}/search", json=payload, timeout=60)
    r.raise_for_status()
    body = r.json()
    assert body["n_inferred"] == 0, "second call should be 100% cache hits"
    assert body["n_cache_hits"] == body["n_retrieved"]
