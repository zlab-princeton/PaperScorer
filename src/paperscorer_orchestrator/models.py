"""Model selection (menu fallback) + download."""
from __future__ import annotations

import os
from pathlib import Path

from rich.console import Console
from rich.prompt import Prompt

from .config import Config, MODALITIES, SIZES, DOMAINS, legacy_model_repo, model_repo, parse_repo

console = Console()


def _multi(title: str, options: list[str], selected: list[str]) -> list[str]:
    """Numbered multiselect for the no-TTY menu fallback: type a space/comma
    list of numbers to toggle the selection."""
    while True:
        console.print(f"\n[bold]{title}[/bold]")
        for i, o in enumerate(options, 1):
            mark = "[green]✓[/green]" if o in selected else " "
            console.print(f"  [{mark}] [cyan]{i}[/cyan] {o}")
        raw = Prompt.ask("  toggle #(s) (space/comma sep) or Enter to accept", default="")
        if not raw.strip():
            if selected:
                return selected
            console.print("  [red]pick at least one[/red]")
            continue
        for tok in raw.replace(",", " ").split():
            if tok.isdigit() and 1 <= int(tok) <= len(options):
                o = options[int(tok) - 1]
                selected = [x for x in selected if x != o] if o in selected else selected + [o]


def model_page(cfg: Config) -> Config:
    """Menu-fallback model chooser (multiselect). Mutates + returns cfg."""
    console.rule("[bold]Model selection")
    console.print("8 fine-tunes: {3b,7b} x {text,vision} x {arxiv,iclr}. "
                  "Pick any combination; one becomes active for run/serve.\n")
    cur = [parse_repo(r) for r in cfg.model.repos] or [(cfg.model.size, cfg.model.modality, cfg.model.domain)]
    sizes = _multi("Size(s)", list(SIZES), list(dict.fromkeys(s for s, _, _ in cur)))
    mods = _multi("Modality(ies)", list(MODALITIES), list(dict.fromkeys(m for _, m, _ in cur)))
    doms = _multi("Domain(s)", list(DOMAINS), list(dict.fromkeys(d for _, _, d in cur)))
    repos = [model_repo(s, m, d) for s in sizes for m in mods for d in doms]
    cfg.model.repos = repos
    active = cfg.model.active if cfg.model.active in repos else repos[0]
    if len(repos) > 1:
        console.print("\n[bold]Active model[/bold] (run/serve)")
        for i, r in enumerate(repos, 1):
            console.print(f"  [cyan]{i}[/cyan] {r}" + ("  [green]← active[/green]" if r == active else ""))
        choice = Prompt.ask("  pick", choices=[str(i) for i in range(1, len(repos) + 1)],
                            default=str(repos.index(active) + 1))
        active = repos[int(choice) - 1]
    cfg.model.active = active
    console.print(f"\n  selected: {', '.join(repos)}\n  active: [bold]{active}[/bold]\n")
    return cfg


def _set_hf_env(cfg: Config) -> None:
    if cfg.cache.hf_home:
        os.environ["HF_HOME"] = cfg.cache.hf_home
        os.environ["HF_HUB_CACHE"] = str(Path(cfg.cache.hf_home) / "hub")


def download_model(cfg: Config, *, revision: str | None = None) -> str:
    """Download ALL selected models into HF_HOME. Returns the active model's path."""
    from huggingface_hub import snapshot_download
    _set_hf_env(cfg)
    active_path = ""
    for repo in cfg.model.repos:
        console.print(f"[bold]Downloading[/bold] {repo} ...")
        p = snapshot_download(repo, revision=revision)
        console.print(f"  [green]✓[/green] {p}")
        if repo == cfg.model.repo:
            active_path = p
    if not active_path:  # active not in repos for some reason -> fetch it
        active_path = snapshot_download(cfg.model.repo, revision=revision)
    return active_path


def download_mineru(cfg: Config) -> None:
    """Download the MinerU models paperprep needs (into HF_HOME).

    Runs paperprep's own ``download_mineru_models()`` inside the paperprep venv
    so it uses paperprep's huggingface_hub + the exact repos it expects.
    """
    import subprocess
    from .setup import PAPERPREP
    pp_py = PAPERPREP / ".venv" / "bin" / "python"
    if not pp_py.exists():
        console.print(f"[yellow]paperprep venv missing — skipping MinerU download[/yellow] ({pp_py})")
        return
    console.print("[bold]Downloading MinerU models[/bold] (paperprep) ...")
    r = subprocess.run(
        [str(pp_py), "-c",
         "from paper_anonymizer.core.ocr import download_mineru_models; "
         "import json; print(json.dumps(download_mineru_models()))"],
        env=cfg.export_env(),
    )
    console.print("  [green]✓ MinerU models[/green]" if r.returncode == 0
                  else f"  [red]✗ MinerU download failed (rc={r.returncode})[/red]")


def download_all(cfg: Config) -> None:
    """Download everything the deployment needs: the selected PaperScorer HF
    model(s) + the MinerU models paperprep needs."""
    download_model(cfg)
    download_mineru(cfg)


def model_is_cached(cfg: Config) -> str | None:
    """Return the ACTIVE model's local snapshot path if cached, else None."""
    from huggingface_hub import snapshot_download
    _set_hf_env(cfg)
    # Snapshots downloaded before the PaperScorer rename sit under the old id.
    for repo in (cfg.model.repo, legacy_model_repo(*parse_repo(cfg.model.repo))):
        try:
            return snapshot_download(repo, local_files_only=True)
        except Exception:
            continue
    return None
