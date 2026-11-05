"""`paperscorer setup` — system-settings page + child-project provisioning.

System settings: cache dirs (HF_HOME, PAPERLENS_DATA_ROOT) + arbitrary env vars.
Child provisioning: each submodule that needs its own venv/conda gets set up
(paperprep -> uv venv with mineru+serve; paperscorer-training-and-inference -> LF venv).
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from rich.console import Console
from rich.prompt import Confirm, Prompt

from .config import Config, ROOT

console = Console()

# Child projects that need provisioning (relative to PaperScorer root).
PAPERPREP = ROOT / "paperprep"
TRAININFER = ROOT / "paperscorer-training-and-inference"
REVIEWING = ROOT / "tools" / "paperscorer-reviewing"
ARXIV_SERVER = ROOT / "tools" / "paperscorer-arxiv-server"


def settings_page(cfg: Config) -> Config:
    """Cache dirs + env vars (key-value editor)."""
    from .ui import edit_kv
    console.rule("[bold]System settings")
    # Prefer a value the user already set: saved config > live $HF_HOME env >
    # HF's own default. People accept the default, so surface the best one.
    default_hf = (cfg.cache.hf_home
                  or os.environ.get("HF_HOME")
                  or str(Path.home() / ".cache" / "huggingface"))
    edited = edit_kv(
        "Cache locations (pick a # to edit, Enter to accept)",
        {"HF_HOME": default_hf, "PAPERLENS_DATA_ROOT": cfg.cache.data_root},
        help={
            "HF_HOME": "model snapshot cache (shared across tools)",
            "PAPERLENS_DATA_ROOT": "reconstructed datasets + page images; blank ok",
        },
    )
    cfg.cache.hf_home = edited["HF_HOME"]
    cfg.cache.data_root = edited["PAPERLENS_DATA_ROOT"]

    # Texlive: real probe (runs `pdflatex --version`) + optional override.
    # LaTeX-dir reviews need pdflatex; this lets users on machines without it
    # on PATH point paperprep at a specific TeX Live install.
    from .texlive import pdflatex_works
    ok, path_pl = pdflatex_works(bin_dir=cfg.paperprep.texlive_bin or None)
    if cfg.paperprep.texlive_bin and ok:
        console.print(f"  texlive_bin (override) works: [bold]{cfg.paperprep.texlive_bin}[/bold]")
    elif cfg.paperprep.texlive_bin and not ok:
        console.print(f"  [red]✗ override set but pdflatex --version fails at "
                      f"{cfg.paperprep.texlive_bin}[/red] — fix path or reinstall")
    elif ok:
        console.print(f"  pdflatex runs: [green]{path_pl}[/green]")
    else:
        console.print("  [red]✗ pdflatex NOT runnable[/red] — LaTeX-dir reviews will fail "
                      "without an override")
    if Confirm.ask("  override texlive bin dir?",
                   default=bool(cfg.paperprep.texlive_bin) or not ok):
        cfg.paperprep.texlive_bin = Prompt.ask(
            "  texlive bin dir (blank = rely on PATH)",
            default=cfg.paperprep.texlive_bin,
        ).strip()

    if Confirm.ask("  edit extra env vars?", default=False):
        console.print("  enter KEY=VALUE lines; blank to finish")
        while True:
            line = Prompt.ask("  env", default="")
            if not line.strip():
                break
            if "=" not in line:
                console.print("    [red]need KEY=VALUE[/red]")
                continue
            k, v = line.split("=", 1)
            cfg.env[k.strip()] = v.strip()
    console.print(f"\n  HF_HOME = [bold]{cfg.cache.hf_home}[/bold]")
    if cfg.cache.data_root:
        console.print(f"  PAPERLENS_DATA_ROOT = [bold]{cfg.cache.data_root}[/bold]")
    if cfg.env:
        console.print(f"  env: {', '.join(f'{k}={v}' for k, v in cfg.env.items())}")
    console.print()
    return cfg


def _have(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def provision_children(cfg: Config, *, dry_run: bool = False, with_retrieval: bool = False) -> None:
    """Set up child project venvs. Builds by default.

    Core (always) — what `paperscorer run` (PDF review) needs:
      paperprep            uv sync --extra gpu --extra serve   (mineru + flask)
      paperscorer-training-and-inference  uv sync             (LF + vLLM scorer)
      paperscorer-reviewing  uv sync                             (UI server)

    Retrieval (only with --with-retrieval) — the search/rerank deployment, which
    is operationally separate and heavier (FAISS indices, 80K corpus, a conda
    env for arxiv_retriever):
      paperscorer-arxiv-server  uv sync
      arxiv_retriever         conda env (separate toolchain; not uv)

    dry_run=True prints the plan and stops. Otherwise the plan is shown, then a
    confirmation gate lets you escape before any (slow) builds run.
    """
    console.rule("[bold]Child provisioning")
    if not _have("uv"):
        console.print("[yellow]warning:[/yellow] `uv` not on PATH; venv creation will be skipped.")

    # `--python 3.11` forces uv to use (and if missing, download) a managed
    # CPython 3.11 instead of reusing whatever `python3` happens to be on PATH.
    # Background: paperprep + traininfer both require 3.11; on nodes whose
    # system python is 3.12 without dev headers, uv silently picks 3.12 and
    # MinerU's triton JIT then fails to compile. Pinning here is cluster-safe
    # because uv-managed pythons ship with full dev headers.
    PY = ["--python", "3.11"]
    plans: list[tuple[str, Path, list[str]]] = []
    if PAPERPREP.exists():
        plans.append(("paperprep (mineru + flask serve)", PAPERPREP,
                      ["uv", "sync", *PY, "--extra", "gpu", "--extra", "serve"]))
    if TRAININFER.exists():
        plans.append(("paperscorer-training-and-inference (LF + vLLM scorer)", TRAININFER,
                      ["uv", "sync", *PY]))
    if REVIEWING.exists():
        # uv sync (not `uv venv`): reviewing is a real uv project (fastapi +
        # uvicorn + ...), so its deps must be installed -- a bare venv would
        # leave the UI server unimportable. uv sync resyncs every run while
        # reusing the existing venv.
        plans.append(("paperscorer-reviewing (UI server)", REVIEWING,
                      ["uv", "sync", *PY]))

    if with_retrieval and ARXIV_SERVER.exists():
        plans.append(("paperscorer-arxiv-server (rerank deployment)", ARXIV_SERVER,
                      ["uv", "sync", *PY]))

    # Guard: a submodule dir exists but is EMPTY when the repo was cloned
    # without --recurse-submodules. `uv sync` there would walk UP to the
    # parent pyproject (this orchestrator) and silently build a bogus root
    # .venv (and even report success). Detect the missing checkout and stop.
    not_init = [(n, p) for (n, p, _) in plans if not (p / "pyproject.toml").exists()]
    if not_init:
        console.print("\n[red]submodules not initialized[/red] — these child dirs are empty "
                      "(repo was cloned without [cyan]--recurse-submodules[/cyan]):")
        for n, p in not_init:
            console.print(f"  • {n}  [dim]{p}[/dim]")
        console.print(
            "\nFrom the repo root, initialize them and re-run [cyan]paperscorer setup[/cyan]:\n"
            f"  [cyan]cd {ROOT}[/cyan]\n"
            "  [cyan]git submodule update --init --recursive[/cyan]\n")
        return

    # `uv sync` re-syncs deps idempotently and REUSES an existing venv (it never
    # recreates it), so we run it on every setup -- keeping deps current with the
    # project's pyproject/lock. `uv venv` only CREATES a venv, so it runs solely
    # when .venv is missing; re-running it would recreate/clobber an existing env.
    def _is_sync(cmd: list[str]) -> bool:
        return cmd[:2] == ["uv", "sync"]

    def _will_run(path: Path, cmd: list[str]) -> bool:
        return _is_sync(cmd) or not (path / ".venv").exists()

    for name, path, cmd in plans:
        if _will_run(path, cmd):
            verb = "re-sync" if _is_sync(cmd) else "create"
            console.print(f"  • {name}: [cyan]{verb}[/cyan]  ({' '.join(cmd)})")
        else:
            console.print(f"  • {name}: .venv [green]exists[/green] → [dim]skip (no recreate)[/dim]")
    if with_retrieval:
        console.print("  • retrieval stack: arxiv-server venv (above) + data/index/config "
                      "checks + arxiv_retriever conda env (provisioned below)")
    else:
        console.print("  [dim]• retrieval stack (arxiv-server + arxiv_retriever conda): "
                      "skipped — pass --with-retrieval[/dim]")

    # What actually runs this invocation: every uv-sync project + any missing venv.
    todo = [(n, p, c) for (n, p, c) in plans if _will_run(p, c)]
    if dry_run:
        console.print("\n  [dim](dry run — nothing run. Drop --dry-run to provision.)[/dim]\n")
        if with_retrieval:
            from .retrieval import provision_retrieval
            provision_retrieval(cfg, dry_run=True)
        return
    if not todo:
        console.print("\n  [green]nothing to do — venvs present, nothing to re-sync.[/green]\n")
        if with_retrieval:
            from .retrieval import provision_retrieval
            provision_retrieval(cfg, dry_run=False)
        return

    # Confirmation gate (escape hatch). `uv sync` is fast when already up to date;
    # a first-time mineru install is ~10 min.
    console.print(f"\n  [yellow]about to provision {len(todo)} project(s)[/yellow] "
                  "(uv sync re-syncs deps; first-time mineru install is ~10 min).")
    from .form import confirm_menu
    if not confirm_menu("provision now?", default_yes=True):
        console.print("  [dim]skipped — re-run `paperscorer setup` when ready.[/dim]\n")
        return

    for name, path, cmd in plans:
        if not _will_run(path, cmd):
            console.print(f"  [dim]skip {name} (.venv exists — no recreate)[/dim]")
            continue
        if not _have("uv"):
            console.print(f"  [red]skip {name}: uv missing[/red]")
            continue
        console.print(f"  {'re-syncing' if _is_sync(cmd) else 'provisioning'} {name} ...")
        r = subprocess.run(cmd, cwd=path)
        console.print(f"  {'[green]✓[/green]' if r.returncode == 0 else f'[red]✗ rc={r.returncode}[/red]'} {name}")
        # traininfer ships a flash-attn wheel pinned to a specific glibc/CUDA
        # combo. Older nodes can install it cleanly but fail at `import
        # flash_attn` (GLIBC_2.32 not found, etc.); vLLM then crashes on
        # model load. Workaround: probe once and uninstall the bad wheel --
        # vLLM falls back to xformers (slower attention, same outputs).
        if r.returncode == 0 and path == TRAININFER:
            _disable_broken_flash_attn(path)

    if with_retrieval:
        from .retrieval import provision_retrieval
        provision_retrieval(cfg, dry_run=False)


def _disable_broken_flash_attn(traininfer_path: Path) -> None:
    """If `import flash_attn` fails in the traininfer venv, uninstall it.

    The pinned wheel (cu12torch2.7cxx11abiFALSE, cp311) installs cleanly via
    `uv sync` but raises ImportError at runtime on nodes with an older glibc
    or mismatched CUDA -- which breaks vLLM startup. Removing flash-attn
    causes vLLM to fall back to xformers (no functionality change, just
    slightly slower attention).
    """
    probe = subprocess.run(
        ["uv", "run", "--", "python", "-c", "import flash_attn"],
        cwd=traininfer_path, capture_output=True, text=True,
    )
    if probe.returncode == 0:
        return
    console.print("  [yellow]flash-attn imports failed in this venv "
                  "(common on older-glibc compute nodes):[/yellow]")
    # Surface the first line of stderr -- usually the GLIBC / undefined-symbol clue
    err = (probe.stderr or "").strip().splitlines()
    if err:
        console.print(f"    [dim]{err[-1]}[/dim]")
    console.print("  [yellow]uninstalling flash-attn so vLLM falls back to "
                  "xformers (slower attention, same outputs)...[/yellow]")
    rm = subprocess.run(
        ["uv", "pip", "uninstall", "flash-attn"],
        cwd=traininfer_path, capture_output=True, text=True,
    )
    if rm.returncode == 0:
        console.print("  [green]✓ flash-attn removed; vLLM will use xformers.[/green]")
    else:
        console.print(f"  [red]✗ uninstall failed (rc={rm.returncode}):[/red]\n"
                      f"    {rm.stderr.strip()}")
