"""``paperscorer`` — top-level orchestrator CLI for the PaperScorer ecosystem.

  paperscorer                 dashboard (config + child/model/gpu status)
  paperscorer setup           wizard: settings -> model -> gpu, then build child venvs
                            (--dry-run to preview the build; --with-retrieval for search stack)
  paperscorer model           model selection page
  paperscorer gpu             gpu / slurm page
  paperscorer config          print the saved .paperscorer.toml
  paperscorer download        snapshot-download the selected HF model
  paperscorer status          check child venvs + model cache
  paperscorer run <pdf>       headless end-to-end review of one PDF
  paperscorer serve           launch the reviewing web UI (long-running, upload PDFs)
"""
from __future__ import annotations

import argparse
import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .config import Config, CONFIG_PATH, LEGACY_CONFIG_PATH, ROOT

console = Console()


def _dashboard(cfg: Config) -> None:
    from .models import model_is_cached
    from .setup import PAPERPREP, TRAININFER, REVIEWING, ARXIV_SERVER

    t = Table(show_header=False, box=None)
    if CONFIG_PATH.exists():
        cfg_line = str(CONFIG_PATH)
    elif LEGACY_CONFIG_PATH.exists():
        # pre-rename file; still read, but `setup` will save the new name
        cfg_line = f"{LEGACY_CONFIG_PATH}  [dim](pre-rename; setup saves {CONFIG_PATH.name})[/dim]"
    else:
        cfg_line = f"{CONFIG_PATH}  [yellow](not saved yet)[/yellow]"
    t.add_row("[bold]config[/bold]", cfg_line)
    n = len(cfg.model.repos)
    t.add_row("[bold]models[/bold]", f"{n} selected" + (f"  ([dim]{', '.join(cfg.model.repos)}[/dim])" if n <= 4 else ""))
    t.add_row("[bold]active[/bold]", f"{cfg.model.repo}")
    cached = model_is_cached(cfg)
    t.add_row("", "[green]✓ active cached[/green]" if cached else "[yellow]active not downloaded[/yellow]  (paperscorer download)")
    t.add_row("[bold]gpu[/bold]", cfg.gpu.mode + (f"  ({cfg.gpu.partition}, {cfg.gpu.time})" if cfg.gpu.mode == "slurm" else ""))
    t.add_row("[bold]HF_HOME[/bold]", cfg.cache.hf_home or "[dim](default)[/dim]")
    t.add_row("[bold]data root[/bold]", cfg.cache.data_root or "[dim](unset)[/dim]")
    console.print(Panel(t, title="PaperScorer", title_align="left"))

    ch = Table(title="children", title_justify="left")
    ch.add_column("project"); ch.add_column(".venv"); ch.add_column("path")
    for name, p in [("paperprep", PAPERPREP), ("paperscorer-training-and-inference", TRAININFER),
                    ("paperscorer-reviewing", REVIEWING), ("paperscorer-arxiv-server", ARXIV_SERVER)]:
        has = (p / ".venv").exists() if p.exists() else None
        mark = "[green]✓[/green]" if has else ("[yellow]missing[/yellow]" if p.exists() else "[red]absent[/red]")
        ch.add_row(name, mark, str(p))
    console.print(ch)
    console.print("\nnext: [cyan]paperscorer setup[/cyan] to configure, then [cyan]paperscorer download[/cyan] + [cyan]paperscorer run <pdf>[/cyan]")


def cmd_setup(args, cfg: Config) -> int:
    from .setup import settings_page, provision_children
    from .models import model_page
    from .gpu import gpu_page
    from .form import run_setup_form

    if getattr(args, "princeton", False):
        # Pre-seed the Princeton slurm defaults so the wizard (or saved config)
        # picks them up. Only overrides values still at their factory blanks --
        # a user who's already set a partition keeps their choice.
        if not cfg.gpu.partition:
            cfg.gpu.partition = "gpu-test"
        if not cfg.gpu.extra_args:
            cfg.gpu.extra_args = "--constraint=gpu80"
        console.print("[dim]  --princeton: seeded gpu.partition=gpu-test, "
                      "extra_args=--constraint=gpu80[/dim]")

    if args.menu:
        res = None                       # force the rich menu pages
    else:
        res = run_setup_form(cfg, with_retrieval=bool(args.with_retrieval))  # None if no TTY
    if res == "__cancel__":
        console.print("[yellow]setup cancelled — nothing saved[/yellow]")
        return 1
    if res is None:
        # fallback: rich menu pages (no TTY, --menu, or textual unavailable)
        cfg = settings_page(cfg)
        cfg = model_page(cfg)
        cfg = gpu_page(cfg)
    else:
        cfg = res
    cfg.save()
    console.print(f"[green]✓ saved[/green] {CONFIG_PATH}")
    provision_children(cfg, dry_run=bool(args.dry_run), with_retrieval=bool(args.with_retrieval))
    if not args.dry_run and not args.no_download:
        from .form import confirm_menu
        from .models import download_all
        console.rule("[bold]Model downloads")
        console.print("  Fetches the selected PaperScorer model(s) + the MinerU models "
                      "paperprep needs, into HF_HOME.")
        if args.serve or confirm_menu("download models now?", default_yes=True):
            download_all(cfg)
        else:
            console.print("  [dim]skipped — run `paperscorer download` for PaperScorer later; "
                          "MinerU also downloads on first paperprep use.[/dim]")
    if args.dry_run:
        return 0
    if args.serve:
        from .models import model_is_cached, download_model
        console.rule("[bold]Serve")
        model_path = model_is_cached(cfg) or download_model(cfg)
        if args.with_retrieval:
            from .workflow import serve_retrieval
            return serve_retrieval(cfg, model_path, port=8000)
        from .workflow import serve_ui
        return serve_ui(cfg, model_path, port=8003)
    # No prompt — setup ends here; tell the user how to launch.
    serve_cmd = "paperscorer serve --with-retrieval" if args.with_retrieval else "paperscorer serve"
    console.print(f"[dim]  Setup complete — run `{serve_cmd}` to launch "
                  f"(or rerun setup with `--serve` to chain straight into the UI).[/dim]")
    return 0


def cmd_model(args, cfg: Config) -> int:
    from .models import model_page
    cfg = model_page(cfg); cfg.save()
    console.print(f"[green]✓ saved[/green] {CONFIG_PATH}")
    return 0


def cmd_gpu(args, cfg: Config) -> int:
    from .gpu import gpu_page
    cfg = gpu_page(cfg); cfg.save()
    console.print(f"[green]✓ saved[/green] {CONFIG_PATH}")
    return 0


def cmd_config(args, cfg: Config) -> int:
    path = CONFIG_PATH if CONFIG_PATH.exists() else LEGACY_CONFIG_PATH
    if not path.exists():
        console.print("[yellow]no config saved yet[/yellow] — run `paperscorer setup`")
        return 0
    if path == LEGACY_CONFIG_PATH:
        console.print(f"[dim]{path}  (pre-rename; setup saves {CONFIG_PATH.name})[/dim]")
    console.print(path.read_text())
    return 0


def cmd_download(args, cfg: Config) -> int:
    from .models import download_model
    download_model(cfg)
    return 0


def cmd_status(args, cfg: Config) -> int:
    _dashboard(cfg)
    return 0


def cmd_serve(args, cfg: Config) -> int:
    from .models import model_is_cached, download_model
    from .workflow import serve_ui, serve_retrieval
    model_path = model_is_cached(cfg)
    if not model_path:
        console.print("[yellow]model not downloaded — fetching now[/yellow]")
        model_path = download_model(cfg)
    if getattr(args, "with_retrieval", False):
        return serve_retrieval(cfg, model_path, port=int(args.port))
    return serve_ui(cfg, model_path, port=int(args.port))


def cmd_run(args, cfg: Config) -> int:
    from pathlib import Path
    from .models import model_is_cached
    from .workflow import run_workflow
    pdf = Path(args.pdf).resolve()
    if not pdf.exists():
        console.print(f"[red]no such file: {pdf}[/red]"); return 2
    model_path = model_is_cached(cfg)
    if not model_path:
        console.print("[yellow]model not downloaded — fetching now[/yellow]")
        from .models import download_model
        model_path = download_model(cfg)
    return run_workflow(cfg, pdf, model_path)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="paperscorer", description="PaperScorer ecosystem orchestrator")
    sub = p.add_subparsers(dest="command", metavar="<command>")
    sp = sub.add_parser("setup", help="configure settings + model + gpu, then build child venvs")
    sp.add_argument("--dry-run", action="store_true",
                    help="show the venv build plan but don't build (default builds, with a confirm gate)")
    sp.add_argument("--with-retrieval", action="store_true",
                    help="also provision the search/rerank stack (arxiv-server venv + arxiv_retriever conda note)")
    sp.add_argument("--menu", action="store_true",
                    help="use the plain numbered-menu editor instead of the interactive walkthrough")
    sp.add_argument("--no-download", action="store_true",
                    help="skip the model-download step (PaperScorer + MinerU) after provisioning")
    sp.add_argument("--serve", action="store_true",
                    help="after provisioning, launch the serve UI directly (combine with "
                         "--with-retrieval for the search UI). Without this, setup ends and "
                         "you launch via `paperscorer serve` separately.")
    sp.add_argument("--princeton", action="store_true",
                    help="seed Princeton-cluster slurm defaults (partition=gpu-test, "
                         "extra_args=--constraint=gpu80) before the wizard runs; safe to "
                         "edit after. Without this, slurm fields start blank and you fill "
                         "them in for your cluster.")
    sp.set_defaults(func=cmd_setup)
    sub.add_parser("model", help="model selection page").set_defaults(func=cmd_model)
    sub.add_parser("gpu", help="gpu / slurm page").set_defaults(func=cmd_gpu)
    sub.add_parser("config", help="print saved config").set_defaults(func=cmd_config)
    sub.add_parser("download", help="download the selected HF model").set_defaults(func=cmd_download)
    sub.add_parser("status", help="dashboard").set_defaults(func=cmd_status)
    rp = sub.add_parser("run", help="headless end-to-end review of one PDF")
    rp.add_argument("pdf")
    rp.set_defaults(func=cmd_run)
    svp = sub.add_parser("serve", help="launch the reviewing web UI (long-running)")
    svp.add_argument("--port", default=8003, help="UI port (default 8003)")
    svp.add_argument("--with-retrieval", action="store_true",
                     help="serve the search/rerank stack (retriever + reranker + search UI) "
                          "instead of the PDF-review UI")
    svp.set_defaults(func=cmd_serve)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = Config.load()
    if not getattr(args, "command", None):
        _dashboard(cfg)
        return 0
    return int(args.func(args, cfg))


if __name__ == "__main__":
    raise SystemExit(main())
