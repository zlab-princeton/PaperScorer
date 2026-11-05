"""Thin HTTP client around the lab's arxiv_retriever /retrieve endpoint.

Tolerates the actual response shape per B.0 exploration:
- Request: {query, topk, return_scores, upper_bound_datetime?, exclude_title?, ...}
- Response: {"result": [[{arxiv_id, contents, title}, ...]]}    (batch-shaped, even for 1 query)
- With return_scores=true the inner dicts become {"document": {...}, "score": float}.

We always ask return_scores=true so the reranker has retriever_score available
for the 0.3*z(retriever) + 0.7*p_accept blend (RANKER.md §5.2).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import requests


log = logging.getLogger(__name__)


# The retriever returns "contents" formatted as the wiki-style stage:
#   '"Title Of The Paper"\nAuthors: ...\nAbstract: ...'
# We parse it lazily on demand via title/abstract properties.
_ABSTRACT_RE = re.compile(r"^Abstract:\s*", re.MULTILINE)


@dataclass
class RetrievedPaper:
    paper_id: str                 # = arxiv_id (canonical key across the ecosystem)
    title: str
    abstract: str
    score: float                  # cosine sim from upstream (0 if not returned)
    contents_raw: str = ""        # the raw "contents" field from arxiv_retriever
    full_text: Optional[str] = None
    submission_date: Optional[str] = None
    # Filled by the server BEFORE handing to the reranker (image_loader.load_pages).
    images: list = field(default_factory=list)


def _split_title_abstract(contents: str) -> tuple[str, str]:
    """Split arxiv_wikiformat 'contents' into (title, abstract).

    Layout (arxiv_wikiformat_per_venue.jsonl):
        "Paper Title"\\nAuthors: ...\\nAbstract: ...

    Return ("", contents) as a safe fallback.
    """
    if not contents:
        return "", ""
    lines = contents.split("\n", 1)
    title = lines[0].strip().strip('"')
    rest = lines[1] if len(lines) > 1 else ""
    # Strip an "Authors:" line + the "Abstract:" prefix
    m = _ABSTRACT_RE.search(rest)
    abstract = rest[m.end():].strip() if m else rest.strip()
    return title, abstract


class ArxivRetrieverClient:
    def __init__(self, base_url: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def retrieve(
        self,
        query: str,
        topk: int = 200,
        *,
        upper_bound_datetime: Optional[str] = None,
        exclude_title: Optional[str] = None,
    ) -> list[RetrievedPaper]:
        payload: dict[str, Any] = {
            "query": query,
            "topk": topk,
            "return_scores": True,
        }
        if upper_bound_datetime is not None:
            payload["upper_bound_datetime"] = upper_bound_datetime
        if exclude_title is not None:
            payload["exclude_title"] = exclude_title

        url = f"{self.base_url}/retrieve"
        log.info(f"POST {url} topk={topk} upper={upper_bound_datetime}")
        r = requests.post(url, json=payload, timeout=self.timeout)
        r.raise_for_status()
        body = r.json()
        # The lab server returns {"result": [[hit, hit, ...]]} (2-level for batch).
        # Tolerate variants from older builds.
        result = body.get("result") if isinstance(body, dict) else body
        if result and isinstance(result, list) and result and isinstance(result[0], list):
            hits = result[0]
        else:
            hits = result or []
        return [self._row_to_paper(row) for row in hits]

    @staticmethod
    def _row_to_paper(row: dict) -> RetrievedPaper:
        # With return_scores=true the row is {document: {...}, score: float}.
        # With return_scores=false the row IS the document.
        if "document" in row and "score" in row:
            doc = row["document"]
            score = float(row["score"])
        else:
            doc = row
            score = float(row.get("score", 0.0))
        contents = str(doc.get("contents", ""))
        title = str(doc.get("title", "")).strip()
        if not title:
            title, _ = _split_title_abstract(contents)
        # We always parse contents for the abstract because the lab server
        # ships title and contents but not a separate abstract field.
        _t, abstract = _split_title_abstract(contents)
        arxiv_id = str(doc.get("arxiv_id") or doc.get("id") or doc.get("paper_id", ""))
        return RetrievedPaper(
            paper_id=arxiv_id,
            title=title,
            abstract=abstract,
            score=score,
            contents_raw=contents,
            full_text=doc.get("full_text") or doc.get("body"),
            submission_date=doc.get("submission_date") or doc.get("date"),
        )
