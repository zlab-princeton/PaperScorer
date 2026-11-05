"""
Simple local PythonInterpreter that uses exec() instead of SandboxFusion.

No external service needed — runs code in-process with stdout capture.
Registered as 'PythonInterpreter' to replace the DR default.
"""

import io
import contextlib
import traceback
from typing import Optional, Union

from qwen_agent.tools.base import BaseTool, register_tool


@register_tool("PythonInterpreter", allow_overwrite=True)
class LocalPythonInterpreter(BaseTool):
    name = "PythonInterpreter"
    description = (
        "Execute Python code locally. Use print() for any output you want to see. "
        "Code must be provided within <code></code> XML tags."
    )
    parameters = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    def __init__(self, cfg: Optional[dict] = None):
        super().__init__(cfg)

    def call(self, params: Union[str, dict], files=None, timeout=30, **kwargs) -> str:
        code = params if isinstance(params, str) else ""
        if not code.strip():
            return "[PythonInterpreter Error]: Empty code."

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                glob = {"__builtins__": __builtins__}
                exec(code, glob, glob)
        except Exception:
            stderr_buf.write(traceback.format_exc())

        result = []
        if stdout_buf.getvalue():
            result.append(f"stdout:\n{stdout_buf.getvalue()}")
        if stderr_buf.getvalue():
            result.append(f"stderr:\n{stderr_buf.getvalue()}")
        return "\n".join(result) if result else "Finished execution."
