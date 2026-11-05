"""FastAPI server: POST /search -> retrieve 200 -> rerank with cached p_accept -> top-k.

Flow per RANKER.md and user-confirmed deployment spec:

    user -> POST /search {query, k<=200}
      retriever.retrieve(query, topk=200)            # always 200 from upstream
      cache.get(ckpt_id, [arxiv_id for _ in 200])    # partition into hit / miss
      reranker.score(papers_miss)                    # vLLM vision inference on misses only
      cache.put(ckpt_id, miss_scores)
      blend = 0.3 * z(retriever_score) + 0.7 * p_accept     (RANKER.md §5.2)
      sort by blend, return top-k
"""
from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Optional

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from omegaconf import OmegaConf
from pydantic import BaseModel, Field

from .cache import PAcceptCache, hash_ckpt_id
from .image_loader import resolve_images_root, resolve_page_paths
from .reranker import PaperLensReranker
from .retriever_client import ArxivRetrieverClient, RetrievedPaper


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
log = logging.getLogger("paperlens-arxiv-server")


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    query: str
    k: int = Field(default=10, ge=1, le=200, description="Number of reranked results to return (<=200)")
    rerank_pool: Optional[int] = Field(
        default=None, ge=1, le=200,
        description="How many retrieved papers to rerank. Defaults to retriever.topk_retrieve "
                    "for /search, and search.rerank_pool for /search_compare.",
    )
    upper_bound_datetime: Optional[str] = Field(
        default=None,
        description="ISO date; arxiv_retriever filters to papers submitted on/before this date",
    )
    exclude_title: Optional[str] = Field(
        default=None,
        description="Fuzzy-match title to exclude from results",
    )
    blend: Optional[str] = Field(
        default=None,
        description="'default' = 0.3*z(retriever)+0.7*p_accept (RANKER.md §5.2). "
                    "'p_accept_only' = rank by p_accept alone.",
    )


class RankedPaper(BaseModel):
    paper_id: str                       # = arxiv_id
    title: str
    abstract: str
    p_accept: float                     # softmax([logp_accept, logp_reject]) in [0,1]
    retriever_score: float              # raw upstream score (cosine for 0.6B-pv; BM25 score for bm25-pv)
    blend_score: float                  # final ranking score
    rerank_position: int                # 0-indexed in returned results
    retriever_position: int             # 0-indexed in upstream top-200
    cache_hit: bool                     # True if p_accept came from cache, False if computed fresh
    submission_date: Optional[str] = None


class SearchResponse(BaseModel):
    query: str
    n_retrieved: int
    n_returned: int
    n_cache_hits: int                   # of the pool, how many had a cached p_accept
    n_inferred: int                     # of the pool, how many we ran through vLLM this request
    results: list[RankedPaper]


class CompareResponse(BaseModel):
    """Side-by-side for the search UI: the SAME reranked pool shown two ways."""
    query: str
    pool: int                           # how many papers were retrieved + reranked
    n_cache_hits: int
    n_inferred: int
    base: list[RankedPaper]             # retriever order (top compare_top); rerank_position shows where each moved
    reranked: list[RankedPaper]         # blend/p_accept order (top compare_top); retriever_position shows origin


# ---------------------------------------------------------------------------
# App + globals
# ---------------------------------------------------------------------------

app = FastAPI(title="paperlens-arxiv-server")
_state: dict = {}
_UI_DIR = Path(__file__).resolve().parent / "ui"


@app.get("/")
def root():
    idx = _UI_DIR / "index.html"
    if not idx.exists():
        raise HTTPException(500, f"UI not bundled: {idx}")
    return FileResponse(idx)


def _zscore(xs: list[float]) -> list[float]:
    if not xs:
        return []
    n = len(xs)
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / max(n - 1, 1)
    std = math.sqrt(var) if var > 0 else 1.0
    return [(x - mean) / std for x in xs]


@app.on_event("startup")
def _startup() -> None:
    cfg_path = os.environ.get("PAPERLENS_SERVER_CONFIG", "configs/server.yaml")
    cfg = OmegaConf.load(cfg_path)
    log.info(f"loaded config from {cfg_path}")
    log.info(OmegaConf.to_yaml(cfg))

    _state["cfg"] = cfg
    _state["images_root"] = resolve_images_root(cfg.get("images_root"))
    # ckpt_id is now derived from paperlens_serve's reported ckpt_path,
    # not a local file (we don't load the model in this process).
    _state["retriever"] = ArxivRetrieverClient(
        base_url=cfg.retriever.base_url,
        timeout=cfg.retriever.timeout_seconds,
    )
    _state["reranker"] = PaperLensReranker(
        serve_url=cfg.paperlens_serve.base_url,
        domain=cfg.reranker.domain,
        modality=cfg.reranker.modality,
        timeout_seconds=float(cfg.paperlens_serve.get("timeout_seconds", 600)),
    )
    # Pull the upstream serve's ckpt_path so the cache key matches what
    # paperlens-serve actually loaded (decouples our config from theirs).
    try:
        h = requests.get(f"{cfg.paperlens_serve.base_url}/health", timeout=10).json()
        _state["ckpt_id"] = hash_ckpt_id(h.get("ckpt_path", cfg.paperlens_serve.base_url))
        _state["compute_arch"] = h.get("compute_arch", "unknown")
    except Exception as e:
        log.warning(f"serve /health probe failed at startup: {e}")
        _state["ckpt_id"] = hash_ckpt_id(cfg.paperlens_serve.base_url)
        _state["compute_arch"] = "unknown"

    _state["cache"] = PAcceptCache(cfg.cache.db_path)
    log.info(
        f"startup complete. ckpt_id={_state['ckpt_id']} "
        f"images_root={_state['images_root']} "
        f"compute_arch={_state['compute_arch']} "
        f"cache_rows={_state['cache'].size(_state['ckpt_id'])}"
    )


@app.on_event("shutdown")
def _shutdown() -> None:
    try:
        _state["cache"].close()
    except Exception:
        pass


@app.get("/health")
def health() -> dict:
    cfg = _state["cfg"]
    return {
        "status": "ok",
        "paperlens_serve_url": cfg.paperlens_serve.base_url,
        "ckpt_id": _state["ckpt_id"],
        "compute_arch": _state["compute_arch"],
        "retriever_url": cfg.retriever.base_url,
        "images_root": str(_state["images_root"]),
        "cache_db": cfg.cache.db_path,
        "cache_rows_for_ckpt": _state["cache"].size(_state["ckpt_id"]),
    }


def _rank_pool(req: SearchRequest, pool: int) -> tuple[list[RankedPaper], int, int]:
    """Retrieve `pool` papers, rerank them (cached p_accept + vLLM on misses),
    blend, and return every paper as a RankedPaper carrying BOTH its
    retriever_position (retriever order) and rerank_position (blend order, global
    within the pool). Records are returned in retriever order. Also returns
    (n_cache_hits, n_inferred). Raises HTTPException(502) if the retriever fails.
    """
    cfg = _state["cfg"]
    try:
        papers = _state["retriever"].retrieve(
            query=req.query,
            topk=pool,
            upper_bound_datetime=req.upper_bound_datetime,
            exclude_title=req.exclude_title,
        )
    except Exception as e:
        log.exception("retriever failed")
        raise HTTPException(502, f"arxiv_retriever error: {e}")
    if not papers:
        return [], 0, 0

    # cache hits + misses; score misses via the upstream reranker
    arxiv_ids = [p.paper_id for p in papers]
    cached = _state["cache"].get(_state["ckpt_id"], arxiv_ids)
    miss_papers = [p for p in papers if p.paper_id not in cached]
    log.info(f"retrieved {len(papers)}, cache hits {len(cached)}, misses {len(miss_papers)}")
    if miss_papers:
        for p in miss_papers:
            p.images = resolve_page_paths(p.paper_id, _state["images_root"])
        miss_scores_list = _state["reranker"].score(miss_papers)
        miss_scores = {p.paper_id: s for p, s in zip(miss_papers, miss_scores_list)}
        _state["cache"].put(_state["ckpt_id"], miss_scores,
                            source="online", compute_arch=_state["compute_arch"])
    else:
        miss_scores = {}
    p_accept_by_id: dict[str, float] = {**cached, **miss_scores}

    # blended ranking score
    z = _zscore([p.score for p in papers])
    blend_mode = (req.blend or cfg.search.get("default_blend", "default")).lower()
    if blend_mode == "p_accept_only":
        blends = [p_accept_by_id.get(p.paper_id, 0.5) for p in papers]
    else:
        wr = float(cfg.search.blend_weight_retriever)
        wp = float(cfg.search.blend_weight_p_accept)
        blends = [wr * zi + wp * p_accept_by_id.get(p.paper_id, 0.5)
                  for zi, p in zip(z, papers)]

    # global rerank_position = rank of each paper when the pool is sorted by blend
    order = sorted(range(len(papers)), key=lambda i: -blends[i])
    rerank_pos = {i: rank for rank, i in enumerate(order)}

    records = [
        RankedPaper(
            paper_id=p.paper_id, title=p.title, abstract=p.abstract,
            p_accept=p_accept_by_id.get(p.paper_id, 0.5),
            retriever_score=p.score, blend_score=blends[i],
            rerank_position=rerank_pos[i], retriever_position=i,
            cache_hit=p.paper_id in cached, submission_date=p.submission_date,
        )
        for i, p in enumerate(papers)
    ]
    return records, len(cached), len(miss_papers)


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    cfg = _state["cfg"]
    pool = req.rerank_pool or int(cfg.retriever.topk_retrieve)
    log.info(f"query={req.query!r} k={req.k} pool={pool} blend={req.blend or 'default'}")
    records, hits, inferred = _rank_pool(req, pool)
    if not records:
        return SearchResponse(query=req.query, n_retrieved=0, n_returned=0,
                              n_cache_hits=0, n_inferred=0, results=[])
    ranked = sorted(records, key=lambda r: r.rerank_position)[: req.k]
    return SearchResponse(
        query=req.query, n_retrieved=len(records), n_returned=len(ranked),
        n_cache_hits=hits, n_inferred=inferred, results=ranked,
    )


@app.post("/search_compare", response_model=CompareResponse)
def search_compare(req: SearchRequest) -> CompareResponse:
    """Base vs reranked, side by side. Retrieves + reranks a middle pool
    (search.rerank_pool, default 50) and returns the top compare_top of each
    ordering, so the UI can show how PaperLens reorders the retriever's hits.
    """
    cfg = _state["cfg"]
    pool = req.rerank_pool or int(cfg.search.get("rerank_pool", 50))
    top = int(cfg.search.get("compare_top", 20))
    log.info(f"[compare] query={req.query!r} pool={pool} top={top} blend={req.blend or 'default'}")
    records, hits, inferred = _rank_pool(req, pool)
    base = sorted(records, key=lambda r: r.retriever_position)[:top]
    reranked = sorted(records, key=lambda r: r.rerank_position)[:top]
    return CompareResponse(
        query=req.query, pool=len(records),
        n_cache_hits=hits, n_inferred=inferred,
        base=base, reranked=reranked,
    )


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/server.yaml")
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    args = ap.parse_args()

    os.environ["PAPERLENS_SERVER_CONFIG"] = args.config
    cfg = OmegaConf.load(args.config)
    host = args.host or cfg.server.host
    port = args.port or cfg.server.port

    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
