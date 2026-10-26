"""Pre-flight checks: NVIDIA GPU, ports, upstream HTTP health."""
from __future__ import annotations

import logging
import shutil
import socket
import subprocess
from dataclasses import dataclass
from typing import Optional


log = logging.getLogger(__name__)


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""


def check_nvidia_gpu() -> CheckResult:
    """Return one CheckResult for nvidia-smi presence + at least one visible GPU."""
    nvsmi = shutil.which("nvidia-smi")
    if not nvsmi:
        return CheckResult(
            "nvidia-gpu", False,
            "nvidia-smi not on PATH; paperlens-serve needs a CUDA GPU.",
        )
    try:
        out = subprocess.check_output(
            [nvsmi, "--query-gpu=name,memory.total", "--format=csv,noheader"],
            stderr=subprocess.STDOUT, text=True, timeout=10,
        )
    except subprocess.CalledProcessError as e:
        return CheckResult("nvidia-gpu", False, f"nvidia-smi exited {e.returncode}: {e.output.strip()}")
    except subprocess.TimeoutExpired:
        return CheckResult("nvidia-gpu", False, "nvidia-smi timeout")
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    if not lines:
        return CheckResult("nvidia-gpu", False, "nvidia-smi returned no GPUs")
    return CheckResult("nvidia-gpu", True, f"{len(lines)} GPU(s): {'; '.join(lines)}")


def check_port_free(host: str, port: int) -> CheckResult:
    """True if (host, port) isn't bound by another process."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.5)
    try:
        s.bind((host, port))
        s.close()
        return CheckResult(f"port-{port}-free", True, f"{host}:{port} available")
    except OSError as e:
        return CheckResult(f"port-{port}-free", False, f"{host}:{port} already in use: {e}")


def check_url_health(url: str, timeout: float = 3.0, path: str = "/health") -> CheckResult:
    """GET <url><path> and return ok if 200 + a JSON body.

    paperlens-serve exposes /health (FastAPI), paperprep-serve exposes /healthz
    (Flask), so callers pass the right path.
    """
    import requests
    full = url.rstrip("/") + path
    try:
        r = requests.get(full, timeout=timeout)
        if r.status_code == 200:
            try:
                return CheckResult(f"{full}", True, str(r.json()))
            except Exception:
                return CheckResult(f"{full}", True, r.text[:120])
        return CheckResult(f"{full}", False, f"HTTP {r.status_code}")
    except Exception as e:
        return CheckResult(f"{full}", False, f"{e}")
