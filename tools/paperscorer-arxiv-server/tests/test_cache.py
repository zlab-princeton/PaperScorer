"""Unit tests for PAcceptCache. Pure-Python (no GPU, no vLLM)."""
from __future__ import annotations

import tempfile
from pathlib import Path

from paperlens_arxiv_server.cache import PAcceptCache, hash_ckpt_id


def test_hash_ckpt_id_stable():
    assert hash_ckpt_id("/a/b/ckpt") == hash_ckpt_id("/a/b/ckpt")
    assert hash_ckpt_id("/a/b/ckpt") != hash_ckpt_id("/a/b/other")
    assert len(hash_ckpt_id("/x")) == 12


def test_put_get_roundtrip(tmp_path: Path):
    cache = PAcceptCache(tmp_path / "p.sqlite")
    ckpt = "abc123"
    n = cache.put(ckpt, {"2305.00001": 0.81, "2305.00002": 0.42})
    assert n == 2
    got = cache.get(ckpt, ["2305.00001", "2305.00002", "2305.00003"])
    assert got == {"2305.00001": 0.81, "2305.00002": 0.42}     # miss=2305.00003


def test_ckpt_isolation(tmp_path: Path):
    cache = PAcceptCache(tmp_path / "p.sqlite")
    cache.put("ckpt_a", {"x": 0.1})
    cache.put("ckpt_b", {"x": 0.9})
    assert cache.get("ckpt_a", ["x"]) == {"x": 0.1}
    assert cache.get("ckpt_b", ["x"]) == {"x": 0.9}


def test_size_counters(tmp_path: Path):
    cache = PAcceptCache(tmp_path / "p.sqlite")
    cache.put("ckpt_a", {f"id{i}": 0.5 for i in range(7)})
    cache.put("ckpt_b", {"only_one": 0.5})
    assert cache.size() == 8
    assert cache.size("ckpt_a") == 7
    assert cache.size("ckpt_b") == 1


def test_bootstrap_skips_existing(tmp_path: Path):
    """Bootstrap must not overwrite already-cached online scores."""
    try:
        import pandas as pd
    except ImportError:
        return  # skip if pandas not installed

    cache = PAcceptCache(tmp_path / "p.sqlite")
    cache.put("ckpt", {"shared_id": 0.99}, source="online", compute_arch="gputest_gpu80")  # already there

    pq_path = tmp_path / "p.parquet"
    pd.DataFrame({
        "arxiv_id": ["shared_id", "new_id"],
        "p_accept_3b": [0.10, 0.42],
    }).to_parquet(pq_path)

    inserted = cache.bootstrap_from_parquet(pq_path, ckpt_id="ckpt")
    assert inserted == 1                              # only new_id inserted
    assert cache.get("ckpt", ["shared_id"]) == {"shared_id": 0.99}    # preserved
    assert cache.get("ckpt", ["new_id"]) == {"new_id": 0.42}


def test_compute_arch_tagged(tmp_path: Path):
    """Cache must record which hardware produced each score for auditability."""
    cache = PAcceptCache(tmp_path / "p.sqlite")
    cache.put("ckpt", {"a": 0.1}, source="online", compute_arch="gputest_gpu80")
    cache.put("ckpt", {"b": 0.2}, source="bootstrap", compute_arch="pli_h100_sxm")
    rows = list(cache.conn.execute(
        "SELECT arxiv_id, source, compute_arch FROM p_accept WHERE reranker_ckpt_id='ckpt' ORDER BY arxiv_id"
    ))
    assert rows == [("a", "online", "gputest_gpu80"), ("b", "bootstrap", "pli_h100_sxm")]
