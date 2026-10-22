"""``paperlens serve`` — FastAPI /score server.

Loads one ``Scorer`` (vLLM + LF tokenizer + qwen2_vl template) at startup,
then exposes:

  POST /score   {papers: [sharegpt_row, ...]} -> {scores: [...]}
  GET  /health  introspection (ckpt, template, calls_served, ...)
  GET  /info    full effective config

Idempotency: scores match scripts/vllm_infer.py + the RANKER.md §6.1
parquet on the same hardware (verified by
``scripts/tests/test_serve_idempotency.py``). Across GPU classes the
absolute values drift ~0.02 mean / 0.21 worst-case at the decision
boundary -- see tools/paperlens-arxiv-server/README.md for the full
distribution and the cache_arch column that audits this.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Optional

from pydantic import BaseModel, Field


log = logging.getLogger(__name__)


# -------- Request / response shapes (module-level so FastAPI introspects
# them as request bodies, not query params -- learned the hard way:
# inner-function class defs cause FastAPI to fall back to query-param
# parsing and return 422 'Field required' for req).

class ScoreRequest(BaseModel):
    papers: list[dict] = Field(
        ...,
        description=(
            "Sharegpt-shaped rows. Each row must contain `conversations` "
            "([{from: system|human|gpt, value: ...}, ...]). Optional: "
            "`images` (list of paths or PIL bytes), `_metadata`."
        ),
    )


class PaperScore(BaseModel):
    p_accept: float
    logp_accept: Optional[float] = None
    logp_reject: Optional[float] = None
    pred: Optional[str] = None


class ScoreResponse(BaseModel):
    scores: list[PaperScore]


def _build_app(cfg):
    """Construct the FastAPI app with cfg + Scorer captured in closures."""
    from fastapi import FastAPI, HTTPException

    from .scoring import Scorer

    app = FastAPI(title="paperlens-serve")
    state: dict = {}

    @app.on_event("startup")
    def _startup() -> None:
        log.info("[serve] startup: building Scorer ...")
        state["scorer"] = Scorer(cfg)
        log.info("[serve] startup complete")

    @app.post("/score", response_model=ScoreResponse)
    def score(req: ScoreRequest) -> ScoreResponse:
        if "scorer" not in state:
            raise HTTPException(503, "scorer not initialized yet")
        if not req.papers:
            return ScoreResponse(scores=[])
        try:
            results = state["scorer"].score(req.papers)
        except Exception as e:
            log.exception("scoring failed")
            raise HTTPException(500, f"scoring error: {e}")
        return ScoreResponse(scores=[PaperScore(**r) for r in results])

    @app.get("/health")
    def health() -> dict:
        if "scorer" not in state:
            return {"status": "initializing"}
        return state["scorer"].health()

    @app.get("/info")
    def info() -> dict:
        from omegaconf import OmegaConf
        return OmegaConf.to_container(cfg, resolve=True)

    return app


def run(args: argparse.Namespace) -> int:
    """Entry point for ``paperlens serve --config ...``."""
    try:
        import uvicorn
        from omegaconf import OmegaConf
    except ImportError as e:
        print(f"ERROR: serve deps missing: {e}", file=sys.stderr)
        print("Install with: uv pip install fastapi uvicorn omegaconf", file=sys.stderr)
        return 2

    cfg_path = args.config
    if not os.path.exists(cfg_path):
        print(f"ERROR: config not found: {cfg_path}", file=sys.stderr)
        return 2

    cfg = OmegaConf.load(cfg_path)
    host = args.host or cfg.server.host
    port = args.port or int(cfg.server.port)

    log_level = str(cfg.get("logging", {}).get("level", "INFO")).lower()
    logging.basicConfig(
        level=log_level.upper(),
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    log.info(f"[serve] config: {cfg_path}")
    log.info(f"[serve] binding: {host}:{port}")

    app = _build_app(cfg)
    uvicorn.run(app, host=host, port=port, log_level=log_level)
    return 0
