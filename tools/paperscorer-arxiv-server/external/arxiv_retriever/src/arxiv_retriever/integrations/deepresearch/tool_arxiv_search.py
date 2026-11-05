"""
qwen-agent tool that wraps the ArxivClient for use inside a ReAct agent
(e.g. Alibaba-NLP/Tongyi-DeepResearch-30B-A3B via MultiTurnReactAgent).

Per-task state (year cutoff + paper title to exclude) is stored in thread-local
state. The hosting runner MUST call `set_task_config(...)` before invoking the
agent on a task; otherwise the tool raises RuntimeError, preventing silent
no-lookahead-bias violations.
"""

import json
import threading
from typing import Optional, Union

from qwen_agent.tools.base import BaseTool, register_tool

from arxiv_retriever.clients.arxiv_client import ArxivClient


_task_config = threading.local()


def set_task_config(upper_bound_datetime: str, exclude_title: str) -> None:
    """Set the no-lookahead constraints for the current thread.

    Must be called immediately before the DeepResearch agent runs on a task.
    Both arguments are required — pass empty strings only if you have a
    very good reason, and the server will then apply no filtering.
    """
    _task_config.cfg = {
        "upper_bound_datetime": upper_bound_datetime,
        "exclude_title": exclude_title,
    }


def clear_task_config() -> None:
    _task_config.cfg = None


@register_tool("arxiv_search", allow_overwrite=True)
class ArxivSearchTool(BaseTool):
    name = "arxiv_search"
    description = (
        "Search arXiv for papers semantically related to a query. "
        "Results are strictly filtered to papers published before the per-review cutoff date "
        "(no-lookahead bias), and the paper being reviewed is automatically excluded. "
        "Use this instead of web search to find related work."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": ["string", "array"],
                "items": {"type": "string"},
                "description": "One or more descriptive semantic queries. Can be a single string or an array of strings.",
            },
        },
        "required": ["query"],
    }

    def __init__(self, cfg: Optional[dict] = None):
        super().__init__(cfg)
        cfg = cfg or {}
        self.client = ArxivClient(
            search_url=cfg.get("search_url", "http://127.0.0.1:8000/retrieve"),
            topk=cfg.get("topk", 5),
            timeout=cfg.get("timeout", 30),
            log_requests=cfg.get("log_requests", False),
        )

    def call(self, params: Union[str, dict], **kwargs) -> str:
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except json.JSONDecodeError:
                return "[arxiv_search] Invalid request: params must be a JSON object with a 'query' field."

        try:
            query = params["query"]
        except (KeyError, TypeError):
            return "[arxiv_search] Invalid request: missing 'query' field."

        cfg = getattr(_task_config, "cfg", None)
        if cfg is None:
            raise RuntimeError(
                "ArxivSearchTool: no per-task config set. "
                "Call set_task_config(upper_bound_datetime, exclude_title) before running the agent."
            )

        upper = cfg.get("upper_bound_datetime")
        exclude = cfg.get("exclude_title")

        # Support both single string and array of queries (DR model often uses array syntax)
        if isinstance(query, list):
            results = []
            for i, q in enumerate(query):
                if not isinstance(q, str) or not q.strip():
                    continue
                result = self.client.semantic_search(
                    query=q.strip(),
                    upper_bound_datetime=upper,
                    exclude_title=exclude,
                )
                results.append(f"## Query {i+1}: \"{q.strip()}\"\n{result}")
            return "\n\n=======\n\n".join(results) if results else "No valid queries provided."

        return self.client.semantic_search(
            query=query,
            upper_bound_datetime=upper,
            exclude_title=exclude,
        )
