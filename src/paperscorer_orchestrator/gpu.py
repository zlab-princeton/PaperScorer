"""GPU / SLURM configuration page."""
from __future__ import annotations

from rich.console import Console
from rich.prompt import Prompt

from .config import Config
from .ui import edit_kv

console = Console()


def gpu_page(cfg: Config) -> Config:
    """Interactive GPU chooser. Mutates + returns cfg."""
    console.rule("[bold]GPU / compute")
    console.print("Where does the model run?\n"
                  "  [cyan]local[/cyan]  — this host already has a visible CUDA GPU (nvidia-smi works)\n"
                  "  [cyan]slurm[/cyan]  — submit a job; edit the fields below\n")
    cfg.gpu.mode = Prompt.ask("  mode", choices=["local", "slurm"], default=cfg.gpu.mode)
    if cfg.gpu.mode == "slurm":
        edited = edit_kv(
            "SLURM options (pick a # to edit, Enter to accept)",
            {
                "partition":  cfg.gpu.partition,
                "time":       cfg.gpu.time or "1:00:00",
                "mem":        cfg.gpu.mem or "64G",
                "constraint": cfg.gpu.constraint,
                "extra_args": cfg.gpu.extra_args,
            },
            help={
                "partition":  "REQUIRED — your cluster's GPU partition (e.g. gpu, ailab)",
                "time":       "HH:MM:SS walltime",
                "mem":        "host RAM (model load needs headroom), e.g. 64G",
                "constraint": "blank for none; cluster-specific (e.g. gpu80 for 80GB A100)",
                "extra_args": "blank for none; e.g. --account=… --qos=… --constraint=…",
            },
        )
        cfg.gpu.partition = edited["partition"]
        cfg.gpu.time = edited["time"]
        cfg.gpu.mem = edited["mem"]
        cfg.gpu.constraint = edited["constraint"]
        cfg.gpu.extra_args = edited["extra_args"]
    console.print(f"\n  gpu -> [bold]{cfg.gpu.mode}[/bold]"
                  + (f"  ({cfg.gpu.partition}, {cfg.gpu.time}, {cfg.gpu.mem}"
                     + (f", {cfg.gpu.constraint}" if cfg.gpu.constraint else "")
                     + (f", {cfg.gpu.extra_args}" if cfg.gpu.extra_args else "") + ")"
                     if cfg.gpu.mode == "slurm" else "") + "\n")
    return cfg


def srun_prefix(cfg: Config) -> list[str]:
    """Build the srun argv prefix for slurm mode (empty list for local)."""
    if cfg.gpu.mode != "slurm":
        return []
    argv = ["srun", "--gres=gpu:1", f"--partition={cfg.gpu.partition}",
            f"--time={cfg.gpu.time}", f"--mem={cfg.gpu.mem}"]
    if cfg.gpu.constraint:
        argv.append(f"--constraint={cfg.gpu.constraint}")
    if cfg.gpu.extra_args:
        argv += cfg.gpu.extra_args.split()
    return argv
