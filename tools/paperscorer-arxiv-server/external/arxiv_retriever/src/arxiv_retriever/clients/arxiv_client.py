"""
Plain-Python client for the arxiv_retriever FAISS server.

Ported from SkyRLSearchEnvs/skyrl-train/skyrl-gym/skyrl_gym/tools/search_arxiv.py
without the skyrl_gym @tool decorator / ToolGroup dependency.

The client supports two hard no-lookahead-bias guarantees, enforced server-side:
  - upper_bound_datetime: only return papers published on/before this ISO date
  - exclude_title:        drop any result whose title matches the paper being reviewed
"""

import json
import logging
import threading
import time
import uuid
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 30
DEFAULT_TOPK = 5
MAX_RETRIES = 10
INITIAL_RETRY_DELAY = 1


def _make_api_request(
    url: str,
    payload: Dict[str, Any],
    headers: Dict[str, str],
    timeout: int,
    log_requests: bool,
    log_prefix: str,
    session: Optional[requests.Session] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Common API request logic with retry on 5xx / connection errors."""
    if session is None:
        session = requests.Session()
        should_close_session = True
    else:
        should_close_session = False

    last_error: Optional[str] = None
    for attempt in range(MAX_RETRIES):
        try:
            if log_requests:
                logger.info(f"{log_prefix}Attempt {attempt + 1}/{MAX_RETRIES}: POST {url}")
            start_time = time.time()
            response = session.post(url, headers=headers, json=payload, timeout=timeout)
            end_time = time.time()

            if response.status_code in (500, 502, 503, 504):
                last_error = (
                    f"{log_prefix}Server error {response.status_code} on "
                    f"attempt {attempt + 1}/{MAX_RETRIES}"
                )
                logger.warning(last_error)
                if attempt < MAX_RETRIES - 1:
                    time.sleep(INITIAL_RETRY_DELAY * (attempt + 1))
                continue

            response.raise_for_status()

            if log_requests:
                logger.info(
                    f"{log_prefix}Success on attempt {attempt + 1} "
                    f"({end_time - start_time:.1f}s)"
                )
            if should_close_session:
                session.close()
            return response.json(), None

        except requests.exceptions.ConnectionError as e:
            last_error = f"{log_prefix}Connection error: {e}"
            logger.warning(last_error)
            if attempt < MAX_RETRIES - 1:
                time.sleep(INITIAL_RETRY_DELAY * (attempt + 1))
            continue
        except requests.exceptions.Timeout as e:
            last_error = f"{log_prefix}Timeout error: {e}"
            logger.warning(last_error)
            if attempt < MAX_RETRIES - 1:
                time.sleep(INITIAL_RETRY_DELAY * (attempt + 1))
            continue
        except requests.exceptions.RequestException as e:
            last_error = f"{log_prefix}Request error: {e}"
            break
        except json.JSONDecodeError as e:
            raw = response.text if "response" in locals() else "N/A"
            last_error = f"{log_prefix}JSON decode error: {e}, response: {raw[:200]}"
            break
        except Exception as e:
            last_error = f"{log_prefix}Unexpected error: {e}"
            break

    logger.error(f"{log_prefix}Request failed after {MAX_RETRIES} attempts: {last_error}")
    if should_close_session:
        session.close()
    return None, last_error


class ArxivClient:
    """
    Thread-safe client for the arxiv_retriever `/retrieve` endpoint.

    Uses a process-wide shared requests.Session per base URL for connection pooling.
    """

    _session_pool: Dict[str, requests.Session] = {}
    _session_lock = threading.Lock()

    @classmethod
    def _get_shared_session(cls, base_url: str) -> requests.Session:
        with cls._session_lock:
            if base_url not in cls._session_pool:
                session = requests.Session()
                adapter = requests.adapters.HTTPAdapter(
                    pool_connections=20,
                    pool_maxsize=20,
                    max_retries=0,
                    pool_block=False,
                )
                session.mount("http://", adapter)
                session.mount("https://", adapter)
                cls._session_pool[base_url] = session
            return cls._session_pool[base_url]

    def __init__(
        self,
        search_url: str = "http://127.0.0.1:8000/retrieve",
        topk: int = DEFAULT_TOPK,
        timeout: int = DEFAULT_TIMEOUT,
        log_requests: bool = True,
    ):
        self.search_url = search_url
        self.topk = topk
        self.timeout = timeout
        self.log_requests = log_requests
        parsed = urlparse(self.search_url)
        self.base_url = f"{parsed.scheme}://{parsed.netloc}"
        self.session = self._get_shared_session(self.base_url)

    def semantic_search(
        self,
        query: str,
        upper_bound_datetime: Optional[str] = None,
        exclude_title: Optional[str] = None,
        topk: Optional[int] = None,
    ) -> str:
        """
        Run a semantic search. Returns a formatted string of top-k results (one per line).

        Args:
            query: semantic query text.
            upper_bound_datetime: ISO date (YYYY-MM-DD). Only returns papers on/before this date.
            exclude_title: title to filter out (the paper being reviewed).
            topk: override default topk.
        """
        if not query or not query.strip():
            return json.dumps({"result": "Empty query provided."})

        request_id = str(uuid.uuid4())
        log_prefix = f"[Semantic Search {request_id}] "

        payload: Dict[str, Any] = {
            "query": query.strip(),
            "topk": topk or self.topk,
            "return_scores": False,
        }
        if upper_bound_datetime:
            payload["upper_bound_datetime"] = upper_bound_datetime
        if exclude_title:
            payload["exclude_title"] = exclude_title

        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        api_response, error_msg = _make_api_request(
            self.search_url, payload, headers, self.timeout, self.log_requests, log_prefix, self.session,
        )
        return self._format_response(api_response, error_msg, "Semantic search")

    def author_search(
        self,
        authors: str,
        upper_bound_datetime: Optional[str] = None,
        exclude_title: Optional[str] = None,
        topk: Optional[int] = None,
    ) -> str:
        """
        Search by comma-separated author last names.
        """
        if not authors or not authors.strip():
            return json.dumps({"result": "Empty author list provided."})

        request_id = str(uuid.uuid4())
        log_prefix = f"[Author Search {request_id}] "

        payload: Dict[str, Any] = {
            "last_name_author_search": authors.strip(),
            "topk": topk or self.topk,
            "return_scores": False,
        }
        if upper_bound_datetime:
            payload["upper_bound_datetime"] = upper_bound_datetime
        if exclude_title:
            payload["exclude_title"] = exclude_title

        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        api_response, error_msg = _make_api_request(
            self.search_url, payload, headers, self.timeout, self.log_requests, log_prefix, self.session,
        )
        return self._format_response(api_response, error_msg, "Author search")

    def _format_response(
        self,
        api_response: Optional[Dict[str, Any]],
        error_msg: Optional[str],
        search_type: str,
    ) -> str:
        if error_msg:
            return f"Search error: {error_msg}"
        if not api_response:
            return "Search request failed or timed out after retries."
        if "error" in api_response:
            return f"Search error: {api_response['error']}"

        raw_results = api_response.get("result", [])
        if not raw_results:
            return "No search results found."

        formatted_docs = []
        for retrieval in raw_results:
            for doc_item in retrieval:
                if "document" in doc_item:
                    content = doc_item["document"]["contents"].strip()
                else:
                    content = doc_item.get("contents", "").strip()
                formatted_docs.append(f"[{len(formatted_docs) + 1}] {content}")

        if not formatted_docs:
            return "No search results found."
        return "\n\n".join(formatted_docs)
