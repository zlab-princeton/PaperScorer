"""PaperLens reranker -- HTTP client to ``paperlens serve``.

This module used to load vLLM in-process. It now delegates all scoring to
the persistent ``paperlens serve`` HTTP endpoint (typically running on a
different host/GPU node). Benefits:

  * paperlens-arxiv-server has no GPU / vLLM / transformers / torch deps.
  * Tokenization parity: scores come from the same LF ``get_dataset(...)``
    path that produced RANKER.md §6.1's predictions_3b.parquet and that
    ``scripts/vllm_infer.py`` uses for offline batches.
  * Independent scaling: the orchestration server (this one) is light;
    only the serve box needs a GPU.

See paperlens-training-and-inference/src/paperlens_cli/serve.py for the
upstream ``/score`` contract.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import requests

from .retriever_client import RetrievedPaper


log = logging.getLogger(__name__)


@dataclass
class PaperScore:
    """One row's worth of /score output."""
    p_accept: float
    logp_accept: Optional[float] = None
    logp_reject: Optional[float] = None
    pred: Optional[str] = None


class PaperLensReranker:
    """Thin HTTP wrapper around ``paperlens serve``'s POST /score endpoint.

    No model weights, no vLLM. ``score(papers)`` builds the sharegpt rows
    that the upstream serve expects, POSTs them, returns floats in [0, 1]
    (the renormalized 2-token softmax over Accept/Reject at the boxed slot).
    """

    def __init__(
        self,
        serve_url: str,
        *,
        domain: str = "arxiv",
        modality: str = "vision",
        timeout_seconds: float = 600.0,
        # Kept for source-compatibility with the in-process callsite.
        # The serve owns the actual ckpt/template/etc.; these are ignored.
        ckpt_path: Optional[str] = None,
        template: Optional[str] = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.7,
        max_model_len: int = 24576,
    ):
        self.serve_url = serve_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.domain = domain.lower()
        self.modality = modality.lower()
        if self.domain not in ("arxiv", "iclr"):
            raise ValueError(f"domain must be arxiv|iclr, got {domain!r}")
        if self.modality not in ("text", "vision"):
            raise ValueError(f"modality must be text|vision, got {modality!r}")

        # We don't load anything; just probe the serve is reachable so
        # configuration errors surface at startup, not first request.
        try:
            r = requests.get(f"{self.serve_url}/health", timeout=10)
            r.raise_for_status()
            h = r.json()
            log.info(f"[reranker] connected to paperlens serve: {h}")
            self.compute_arch = h.get("compute_arch", "unknown")
        except Exception as e:
            log.warning(
                f"[reranker] /health probe failed at {self.serve_url}: {e}. "
                "Will retry on first /score call."
            )
            self.compute_arch = "unknown"

    def score(self, papers: list[RetrievedPaper]) -> list[float]:
        """Return p_accept in [0, 1] per paper.

        Builds the same sharegpt rows that ``scripts/reconstruction.py``
        emits (so the upstream serve's LF tokenizer sees the same input
        the model was trained on).
        """
        if not papers:
            return []
        rows = [self._paper_to_sharegpt(p) for p in papers]
        body = {"papers": rows}
        url = f"{self.serve_url}/score"
        log.info(f"[reranker] POST {url} n={len(papers)}")
        resp = requests.post(url, json=body, timeout=self.timeout_seconds)
        resp.raise_for_status()
        scores_raw = resp.json()["scores"]
        if len(scores_raw) != len(papers):
            raise RuntimeError(
                f"serve returned {len(scores_raw)} scores for {len(papers)} papers"
            )
        return [float(s["p_accept"]) for s in scores_raw]

    def score_detailed(self, papers: list[RetrievedPaper]) -> list[PaperScore]:
        """Same as score() but returns the full PaperScore (logp_accept,
        logp_reject, pred) per row -- useful for the cache-write path.
        """
        if not papers:
            return []
        rows = [self._paper_to_sharegpt(p) for p in papers]
        body = {"papers": rows}
        resp = requests.post(f"{self.serve_url}/score", json=body, timeout=self.timeout_seconds)
        resp.raise_for_status()
        return [PaperScore(**s) for s in resp.json()["scores"]]

    # ----------------------------------------------------------------------
    # Internal: build sharegpt rows the upstream serve expects
    # ----------------------------------------------------------------------

    def _paper_to_sharegpt(self, paper: RetrievedPaper) -> dict:
        """Compose a sharegpt row that mirrors the training format.

        For vision, ``paper.images`` should already be populated with
        either local PNG paths OR PIL.Image objects (the server-side
        image_loader fills these in).
        """
        from .prompts import (
            SHAREGPT_SYSTEM_PROMPT, PROMPT_ARXIV, PROMPT_ICLR,
        )

        prompt = PROMPT_ARXIV if self.domain == "arxiv" else PROMPT_ICLR

        # Build the human turn body in the canonical paperprep shape.
        title = (paper.title or "").strip()
        abstract = (paper.abstract or "").strip()
        if self.modality == "vision":
            parts = [prompt]
            if title:
                parts.append(f"# {title}")
            if abstract:
                parts.append(f"## Abstract\n{abstract}")
            n_images = len(paper.images) if paper.images else 0
            if n_images > 0:
                parts.append(" ".join(["<image>"] * n_images))
            human_value = "\n\n".join(parts)
        else:
            body = paper.full_text or f"# {title}\n\n## Abstract\n{abstract}"
            human_value = f"{prompt}\n\n{body.strip()}"

        row: dict = {
            "conversations": [
                {"from": "system", "value": SHAREGPT_SYSTEM_PROMPT},
                {"from": "human",  "value": human_value},
                # The LF "ppo" stage requires an assistant turn for the label
                # column even at inference time; serve discards it after
                # tokenization. Use a neutral placeholder.
                {"from": "gpt",    "value": "Outcome: \\boxed{Accept}"},
            ],
            "_metadata": {
                "arxiv_id": paper.paper_id,
                "title": title,
            },
        }
        if self.modality == "vision":
            # paper.images may be PIL Image instances or path strings; we pass
            # whatever was loaded. The upstream LF mm_plugin handles both.
            row["images"] = [
                p if isinstance(p, str) else getattr(p, "filename", None) or p
                for p in (paper.images or [])
            ]
        return row

    def close(self) -> None:
        pass
