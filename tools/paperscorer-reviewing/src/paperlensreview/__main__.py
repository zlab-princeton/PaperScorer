"""``paperlensreview`` CLI -- single subcommand ``serve``.

Usage:
    paperlensreview serve [--config configs/server.yaml] [--port 8003]

On start:
  1. Run pre-flight checks (NVIDIA GPU, port free, paperprep CLI).
  2. Probe the upstream paperlens-serve /health. If not up, print a
     friendly "start paperlens serve first" message and exit.
  3. Bind FastAPI on the configured host:port.

We do NOT start paperlens-serve here -- that's the launcher script's job
(scripts/launch_local.sh). Keeps the CLI's responsibilities tight.
"""
from __future__ import annotations

import argparse
import logging
import sys
from typing import Optional


log = logging.getLogger("paperlensreview-cli")


def _add_serve(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("serve", help="Start the paperlensreview FastAPI server")
    p.add_argument("--config", default="configs/server.yaml",
                   help="Path to the server config YAML")
    p.add_argument("--host", default=None, help="Override server.host")
    p.add_argument("--port", type=int, default=None, help="Override server.port")
    p.add_argument("--skip_preflight", action="store_true",
                   help="Skip GPU + paperlens-serve health checks (debug only)")
    p.set_defaults(func=_cmd_serve)


def _cmd_serve(args: argparse.Namespace) -> int:
    import os
    from omegaconf import OmegaConf
    from .checks import (
        CheckResult, check_nvidia_gpu,
        check_port_free, check_url_health,
    )
    from .server import run_uvicorn

    cfg = OmegaConf.load(args.config)
    # Allow the launcher to override upstream URLs when it picks dynamic ports;
    # without this, the reviewing server would hit stale base_urls.
    if (env := os.environ.get("PAPERLENS_SERVE_BASE_URL")):
        cfg.paperlens_serve.base_url = env
    if (env := os.environ.get("PAPERPREP_SERVE_BASE_URL")):
        cfg.paperprep_serve.base_url = env
    host = args.host or cfg.server.host
    port = args.port or int(cfg.server.port)

    if not args.skip_preflight:
        results: list[CheckResult] = [
            check_nvidia_gpu(),
            check_url_health(cfg.paperlens_serve.base_url, path="/health"),
            check_url_health(cfg.paperprep_serve.base_url, path="/healthz"),
            check_port_free(host, port),
        ]
        fails = [r for r in results if not r.ok]
        for r in results:
            mark = "✓" if r.ok else "✗"
            print(f"  {mark} {r.name:<32s} {r.detail}", file=sys.stderr)
        if fails:
            print(
                "\nPre-flight FAILED. Address the above and retry.\n"
                "Tip: scripts/launch_local.sh starts paperprep-serve + paperlens-serve first.",
                file=sys.stderr,
            )
            return 2

    print(f"\nServing paperlensreview at http://{host}:{port}\n", file=sys.stderr)
    return run_uvicorn(args.config, host=host, port=port)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="paperlensreview",
        description="PaperLens reviewing UI: upload a PDF, get a verdict + p_accept.",
    )
    sub = p.add_subparsers(dest="command", required=True, metavar="<command>")
    _add_serve(sub)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
