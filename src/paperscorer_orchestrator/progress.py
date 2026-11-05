"""Live warmup dashboard for `paperscorer serve` / `run`.

Renders a rich Live panel while the job runs:
  - one status row per service (spinner -> ✓ / ✗), driven by @@PL markers the
    job script emits on stdout (services bind 127.0.0.1 on the compute node, so
    they can't be health-probed from here):
        @@PL <key> ready | @@PL <key> failed | @@PL url <text>
  - a separate scrolling box per daemon log file (tailed from shared scratch)
  - a box for the orchestrator/reviewing stream (the job's own stdout)
"""
from __future__ import annotations

import os
import queue
import signal
import subprocess
import threading
from collections import deque
from pathlib import Path
from typing import Callable

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

console = Console()

MARKER = "@@PL "


def run_with_progress(
    cmd: list[str],
    env: dict,
    services: list[tuple[str, str]],            # [(key, label), ...]
    *,
    log_files: list[tuple[str, Path]] | None = None,   # [(box title, path), ...]
    interrupt: Callable[[subprocess.Popen], None] | None = None,
    on_marker: Callable[[str, str], None] | None = None,   # ``@@PL <key> <value>`` -> (key, value)
    main_lines: int = 8,
    file_lines: int = 10,
) -> int:
    """Run cmd; render status rows + a box per daemon log + the main stream.
    Returns the process return code (130 on Ctrl-C)."""
    log_files = log_files or []
    state: dict[str, str] = {k: "wait" for k, _ in services}
    pinned: list[str] = []
    mainbuf: deque[str] = deque(maxlen=main_lines)
    filebufs: dict[str, deque[str]] = {t: deque(maxlen=file_lines) for t, _ in log_files}
    offsets: dict[str, int] = {t: 0 for t, _ in log_files}

    proc = subprocess.Popen(
        cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, start_new_session=True,
    )
    q: queue.Queue = queue.Queue()

    def _reader():
        for line in proc.stdout:                 # type: ignore[union-attr]
            q.put(line.rstrip("\n"))
        q.put(None)
    threading.Thread(target=_reader, daemon=True).start()

    def _poll_files():
        for title, path in log_files:
            try:
                if not path.exists():
                    continue
                with open(path, "r", errors="replace") as f:
                    f.seek(offsets[title])
                    chunk = f.read()
                    offsets[title] = f.tell()
                for ln in chunk.splitlines():
                    filebufs[title].append(ln)
            except Exception:
                pass

    def _ansi_lines(buf: deque) -> Text | str:
        """Build a Rich Text from raw log lines: parses embedded ANSI escape
        sequences (Flask color codes, vLLM banners, etc.) so colors render
        AND the visible-width calculation is accurate -- otherwise the
        invisible \\x1b[...m bytes count toward width and panel borders walk.
        Also avoids `[...]` in log lines being mis-parsed as Rich markup."""
        if not buf:
            return ""
        return Text("\n").join(Text.from_ansi(ln) for ln in buf)

    def render() -> Group:
        status = Table.grid(padding=(0, 2))
        for k, label in services:
            s = state[k]
            icon: object = (Text("✓", style="bold green") if s == "ready"
                            else Text("✗", style="bold red") if s == "fail"
                            else Spinner("dots", style="cyan"))
            tail = {"ready": "[green]ready[/green]", "fail": "[red]failed[/red]"}.get(s, "starting…")
            status.add_row(icon, f"{label}  {tail}")
        parts: list = [status]
        if pinned:
            parts.append(Text("\n".join(pinned), style="bold cyan"))
        for title, _ in log_files:
            body = _ansi_lines(filebufs[title]) or Text("(no output yet)", style="dim")
            parts.append(Panel(body, title=title, border_style="cyan", height=file_lines + 2))
        main = _ansi_lines(mainbuf) or Text("(starting…)", style="dim")
        parts.append(Panel(main, title="orchestrator / reviewing", border_style="dim", height=main_lines + 2))
        return Group(*parts)

    def _handle(line: str) -> None:
        if line.startswith(MARKER):
            rest = line[len(MARKER):].strip()
            if rest.startswith("url "):
                pinned.append(rest[4:].strip()); return
            # Split into key + value (value can contain spaces, e.g. URLs).
            p = rest.split(None, 1)
            if len(p) == 2 and p[0] in state and p[1] in ("ready", "failed"):
                state[p[0]] = "ready" if p[1] == "ready" else "fail"; return
            # Forward any other `@@PL <key> <value>` to the caller's hook.
            # Used by serve_ui to capture compute-node URLs and launch the
            # reviewing UI on the login node mid-flight.
            if len(p) == 2 and on_marker is not None:
                try:
                    on_marker(p[0], p[1])
                except Exception:
                    pass
                return
        mainbuf.append(line)

    rc = 0
    try:
        with Live(render(), console=console, refresh_per_second=10) as live:
            done = False
            while not done:
                try:
                    item = q.get(timeout=0.15)
                except queue.Empty:
                    _poll_files(); live.update(render()); continue
                if item is None:
                    done = True
                else:
                    _handle(item)
                _poll_files(); live.update(render())
            _poll_files(); live.update(render())
        rc = proc.wait()
    except KeyboardInterrupt:
        console.print("\n[yellow]stopping…[/yellow]")
        if interrupt:
            interrupt(proc)
        else:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGINT)
            except ProcessLookupError:
                pass
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
        rc = 130
    return rc
