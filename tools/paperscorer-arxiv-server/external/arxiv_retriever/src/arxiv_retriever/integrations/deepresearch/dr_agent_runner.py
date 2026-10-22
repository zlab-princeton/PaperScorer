"""
Reusable runner that invokes the Alibaba-NLP Tongyi-DeepResearch ReAct agent
(`MultiTurnReactAgent`) with a restricted tool set:
  - arxiv_search  (no-lookahead-bias filtered, via the arxiv_retriever server)
  - PythonInterpreter  (inherited from DR defaults)

All other DR tools (search, visit, google_scholar, parse_file) are REMOVED so
the agent physically cannot touch the web or read local files.

The DR `react_agent` module builds a module-level `TOOL_MAP` dict at import time
and looks up tools through it directly (rather than honoring `function_list` on
`MultiTurnReactAgent.__init__`). We therefore monkey-patch `TOOL_MAP` after import.
We also override `SYSTEM_PROMPT` with one that only describes the two allowed tools.

Before each task, `set_task_config(upper_bound_datetime, exclude_title)` must be
called on the thread that will run the agent — this sets the no-lookahead
constraints for the `arxiv_search` tool via thread-local state.

Based on the pattern in `DeepResearch/inference/run_multi_react.py:155-225`.
"""

import concurrent.futures
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from arxiv_retriever.integrations.deepresearch.tool_arxiv_search import (
    ArxivSearchTool,
    clear_task_config,
    set_task_config,
)
from arxiv_retriever.integrations.deepresearch.tool_python_local import LocalPythonInterpreter


# ---------------------------------------------------------------------------
# Custom system prompt — only describes `arxiv_search` and `PythonInterpreter`.
# ---------------------------------------------------------------------------
CUSTOM_SYSTEM_PROMPT = """You are an expert peer reviewer for a top ML conference. Your task is to write a thorough, critical review of the paper the user gives you.

You have EXACTLY TWO tools and no others. Do not attempt to call any other function.

# Tools

<tools>
{"type": "function", "function": {"name": "arxiv_search", "description": "Search arXiv for papers semantically related to a query. Results are strictly filtered to papers published before the per-review cutoff date (no-lookahead bias), and the paper being reviewed is automatically excluded. Use this to find related work, check novelty, and verify claims. Accepts a single query string or an array of query strings.", "parameters": {"type": "object", "properties": {"query": {"type": ["string", "array"], "items": {"type": "string"}, "description": "One or more descriptive semantic queries. Can be a single string or an array of strings."}}, "required": ["query"]}}}
{"type": "function", "function": {"name": "PythonInterpreter", "description": "Executes Python code. Pass the code as a string in the 'code' argument. Use print() for any output you want to see.", "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": "Python code to execute. Use print() for output."}}, "required": ["code"]}}}
</tools>

For each function call, return a JSON object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

When you have gathered enough evidence, produce your final answer as a single JSON object enclosed in <answer></answer> tags, matching exactly this schema (all values are strings except "accept" which is a boolean):
{"strengths": "...", "weaknesses": "...", "critical_disputes": "...", "most_important_strength": "...", "most_important_weakness": "...", "relation_to_other_work": "...", "paper_quality": "...", "accept": true/false}

Current date: """


def _install_deepresearch_on_path() -> None:
    """Make the DeepResearch inference directory importable."""
    dr_path = os.environ.get(
        "DEEPRESEARCH_INFERENCE_DIR",
        "/scratch/gpfs/ZHUANGL/sk7524/DeepResearch/inference",
    )
    if dr_path not in sys.path:
        sys.path.insert(0, dr_path)


def _patch_deepresearch_tools(search_url: str) -> None:
    """Replace `react_agent.TOOL_MAP` and `SYSTEM_PROMPT` with our restricted set.

    Must be called exactly once per process, before any agent instantiation.
    """
    _install_deepresearch_on_path()
    import react_agent  # noqa: E402

    # Use our local exec()-based PythonInterpreter instead of DR's
    # SandboxFusion-based one (which requires an external service).
    python_tool = LocalPythonInterpreter()

    # Instantiate our arxiv_search tool pointed at the local server.
    arxiv_tool = ArxivSearchTool(cfg={"search_url": search_url})

    react_agent.TOOL_MAP = {
        "arxiv_search": arxiv_tool,
        "PythonInterpreter": python_tool,
    }
    react_agent.SYSTEM_PROMPT = CUSTOM_SYSTEM_PROMPT
    react_agent.MAX_LLM_CALL_PER_RUN = 50

    # Monkey-patch _run to fix both forced-answer prompts:
    # 1. Token limit: ask for JSON explicitly (original just says "<answer>your answer</answer>")
    # 2. Call limit: force an answer instead of just giving up with "No answer found."
    _original_run = react_agent.MultiTurnReactAgent._run

    FORCED_ANSWER_PROMPT = (
        "You have run out of budget. Stop making tool calls. Based on everything above, "
        "produce your FINAL review NOW as a JSON object inside <answer></answer> tags.\n"
        "Schema (all strings except accept which is bool):\n"
        '<answer>{"strengths": "...", "weaknesses": "...", "critical_disputes": "...", '
        '"most_important_strength": "...", "most_important_weakness": "...", '
        '"relation_to_other_work": "...", "paper_quality": "...", "accept": true/false}</answer>'
    )

    def _patched_run(self, data, model, **kwargs):
        from react_agent import SYSTEM_PROMPT, TOOL_MAP, MAX_LLM_CALL_PER_RUN, today_date
        import json5

        self.model = model
        try:
            question = data["item"]["question"]
        except Exception:
            raw_msg = data["item"]["messages"][1]["content"]
            question = raw_msg.split("User:")[1].strip() if "User:" in raw_msg else raw_msg

        start_time = time.time()
        planning_port = data["planning_port"]
        answer = data["item"]["answer"]
        self.user_prompt = question
        system_prompt = SYSTEM_PROMPT + str(today_date())
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
        num_llm_calls_available = MAX_LLM_CALL_PER_RUN
        round_num = 0

        def _force_answer(messages, reason):
            """Force the agent to produce a JSON answer."""
            messages[-1]["content"] = FORCED_ANSWER_PROMPT
            content = self.call_server(messages, planning_port)
            messages.append({"role": "assistant", "content": content.strip()})
            if "<answer>" in content and "</answer>" in content:
                prediction = content.split("<answer>")[1].split("</answer>")[0]
                termination = f"forced answer ({reason})"
            else:
                prediction = content
                termination = f"format error: forced answer ({reason})"
            return {
                "question": question, "answer": answer,
                "messages": messages, "prediction": prediction,
                "termination": termination,
            }

        while num_llm_calls_available > 0:
            if time.time() - start_time > 150 * 60:
                return {
                    "question": question, "answer": answer, "messages": messages,
                    "prediction": "No answer found after 2h30mins",
                    "termination": "No answer found after 2h30mins",
                }
            round_num += 1
            num_llm_calls_available -= 1
            content = self.call_server(messages, planning_port)
            print(f"Round {round_num}: {content[:200]}")
            if "<tool_response>" in content:
                content = content[: content.find("<tool_response>")]
            messages.append({"role": "assistant", "content": content.strip()})

            if "<tool_call>" in content and "</tool_call>" in content:
                tool_call_str = content.split("<tool_call>")[1].split("</tool_call>")[0]
                try:
                    if "python" in tool_call_str.lower():
                        try:
                            code_raw = None
                            # Method 1: code in JSON arguments (Qwen3.5 native)
                            try:
                                tc = json5.loads(tool_call_str)
                                code_raw = tc.get("arguments", {}).get("code", "")
                            except Exception:
                                pass
                            # Method 2: <code> block after </tool_call> (DR format)
                            if not code_raw:
                                tc_end_pos = content.find("</tool_call>")
                                after_tc = content[tc_end_pos:] if tc_end_pos >= 0 else ""
                                if "<code>" in after_tc and "</code>" in after_tc:
                                    code_raw = after_tc.split("<code>")[1].split("</code>")[0].strip()
                            # Method 3: <code> block inside <tool_call> (original DR format)
                            if not code_raw and "<code>" in tool_call_str and "</code>" in tool_call_str:
                                code_raw = tool_call_str.split("<code>")[1].split("</code>")[0].strip()
                            if not code_raw:
                                raise ValueError("No code found")
                            result = TOOL_MAP["PythonInterpreter"].call(code_raw)
                        except Exception:
                            result = "[Python Interpreter Error]: Formatting error."
                    else:
                        tc = json5.loads(tool_call_str)
                        tool_name = tc.get("name", "")
                        tool_args = tc.get("arguments", {})
                        result = self.custom_call_tool(tool_name, tool_args)
                except Exception:
                    result = 'Error: Tool call is not valid JSON with "name" and "arguments".'
                messages.append({"role": "user", "content": f"<tool_response>\n{result}\n</tool_response>"})

            if "<answer>" in content and "</answer>" in content:
                break

            # Approaching call limit — force answer with 2 calls left as buffer
            if num_llm_calls_available <= 1 and "<answer>" not in content:
                return _force_answer(messages, "call limit")

            token_count = self.count_tokens(messages)
            print(f"round: {round_num}, token count: {token_count}")
            if token_count > 110 * 1024:
                print(f"Token limit exceeded: {token_count}")
                return _force_answer(messages, "token limit")

        if "<answer>" in messages[-1]["content"]:
            prediction = messages[-1]["content"].split("<answer>")[1].split("</answer>")[0]
            termination = "answer"
        else:
            prediction = "No answer found."
            termination = "exceed available llm calls"
        return {
            "question": question, "answer": answer,
            "messages": messages, "prediction": prediction,
            "termination": termination,
        }

    react_agent.MultiTurnReactAgent._run = _patched_run


def run_agent_on_tasks(
    tasks: List[Dict[str, Any]],
    model_path: str,
    search_url: str = "http://127.0.0.1:8000/retrieve",
    workers: int = 8,
    rollouts: int = 1,
    temperature: float = 0.6,
    top_p: float = 0.95,
    presence_penalty: float = 1.1,
    per_task_timeout_seconds: int = 1800,
    on_result_callback: Optional[callable] = None,
) -> List[Dict[str, Any]]:
    """
    Run the DeepResearch agent on a batch of review tasks.

    Each task dict MUST contain:
      - "item": {"question": str, "answer": str, "_submission_id": str,
                 "_year": int, "_title": str}
      - "planning_port": int  (vLLM port to route this task to)

    Before calling the agent on a task, the runner sets thread-local config:
      upper_bound_datetime = f"{item['_year']}-01-01"
      exclude_title        = item['_title']

    Returns: list of agent result dicts, one per (task × rollout). Each has
    the DR-native fields {question, answer, messages, prediction, termination}
    plus {_submission_id, _year, _title, rollout_idx}.
    """
    _patch_deepresearch_tools(search_url=search_url)

    # Import after patching so the patched globals are used.
    import react_agent  # noqa: E402
    from react_agent import MultiTurnReactAgent  # noqa: E402

    llm_cfg = {
        "model": model_path,
        "generate_cfg": {
            "max_input_tokens": 110 * 1024,
            "max_retries": 10,
            "temperature": temperature,
            "top_p": top_p,
            "presence_penalty": presence_penalty,
        },
        "model_type": "qwen_dashscope",
    }

    agent = MultiTurnReactAgent(
        llm=llm_cfg,
        function_list=["arxiv_search", "PythonInterpreter"],
    )

    def _run_one(task: Dict[str, Any], rollout_idx: int) -> Dict[str, Any]:
        item = task["item"]
        year = item["_year"]
        title = item["_title"]
        # Set no-lookahead constraints for this thread BEFORE the agent runs.
        set_task_config(
            upper_bound_datetime=f"{year}-01-01",
            exclude_title=title,
        )
        try:
            result = agent._run(task, model_path)
        finally:
            clear_task_config()
        result["_submission_id"] = item.get("_submission_id", "")
        result["_year"] = year
        result["_title"] = title
        result["rollout_idx"] = rollout_idx
        return result

    results: List[Dict[str, Any]] = []
    all_invocations = [(t, r) for t in tasks for r in range(rollouts)]

    # Write lock for thread-safe incremental JSONL output.
    _write_lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_task = {
            executor.submit(_run_one, task, rollout_idx): (task, rollout_idx)
            for task, rollout_idx in all_invocations
        }
        for future in tqdm(as_completed(future_to_task), total=len(all_invocations), desc="DR agent tasks"):
            task, rollout_idx = future_to_task[future]
            item = task["item"]
            try:
                result = future.result(timeout=per_task_timeout_seconds)
                results.append(result)
            except concurrent.futures.TimeoutError:
                result = {
                    "question": item.get("question", ""),
                    "answer": item.get("answer", ""),
                    "messages": [],
                    "prediction": "[Failed]",
                    "termination": f"timeout (>{per_task_timeout_seconds}s)",
                    "_submission_id": item.get("_submission_id", ""),
                    "_year": item.get("_year"),
                    "_title": item.get("_title", ""),
                    "rollout_idx": rollout_idx,
                    "error": f"timeout (>{per_task_timeout_seconds}s)",
                }
                results.append(result)
            except Exception as exc:
                result = {
                    "question": item.get("question", ""),
                    "answer": item.get("answer", ""),
                    "messages": [],
                    "prediction": "[Failed]",
                    "termination": f"exception: {exc}",
                    "_submission_id": item.get("_submission_id", ""),
                    "_year": item.get("_year"),
                    "_title": item.get("_title", ""),
                    "rollout_idx": rollout_idx,
                    "error": f"exception: {exc}",
                }
                results.append(result)

            # Incremental write: flush each result as it completes so partial
            # results survive SLURM timeouts.
            if on_result_callback is not None:
                with _write_lock:
                    on_result_callback(result)

    return results
