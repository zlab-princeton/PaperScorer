"""submitit wrapper for paperprep — per-stage StageConfig.

Each stage (compile, mineru, normalize, filter, export) gets its own
SLURM resource profile. MinerU goes to the GPU partition; the rest go to
the CPU partition. Per-stage overrides are accepted via `--stage-overrides`
or a YAML config.

Key design choice: paperprep does NOT chain stages via `afterok` SLURM
dependency. The driver runs each stage to completion (waiting on submitit)
before kicking off the next. This is simpler, restartable, and the
state.jsonl file ensures `--resume` works across stage boundaries.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence


@dataclass
class StageConfig:
    """Per-stage SLURM resource profile."""
    partition: str = ""               # filled by submit_stage based on cpu/gpu routing
    timeout_min: int = 60
    cpus_per_task: int = 4
    mem_gb: int = 16
    gpus_per_node: int = 0
    array_parallelism: int = 80
    extra_slurm: dict = field(default_factory=dict)  # passes through to slurm_*


# Production defaults — match the existing arxiv pipeline's resource profiles.
DEFAULT_STAGE_CONFIGS: dict[str, StageConfig] = {
    "compile":   StageConfig(timeout_min=120, cpus_per_task=4, mem_gb=8),
    "mineru":    StageConfig(timeout_min=240, cpus_per_task=4, mem_gb=64,
                             gpus_per_node=1, array_parallelism=20),
    "normalize": StageConfig(timeout_min=120, cpus_per_task=8, mem_gb=16),
    "filter":    StageConfig(timeout_min=30,  cpus_per_task=4, mem_gb=8),
    "export":    StageConfig(timeout_min=30,  cpus_per_task=4, mem_gb=8),
}


def get_stage_config(
    stage_name: str,
    *,
    cpu_partition: str,
    gpu_partition: str,
    overrides: dict | None = None,
) -> StageConfig:
    """Resolve a StageConfig: defaults + partition routing + caller overrides."""
    if stage_name not in DEFAULT_STAGE_CONFIGS:
        raise KeyError(f"unknown stage: {stage_name!r}; known: {sorted(DEFAULT_STAGE_CONFIGS)}")
    cfg = dataclasses.replace(DEFAULT_STAGE_CONFIGS[stage_name])
    cfg.partition = gpu_partition if cfg.gpus_per_node > 0 else cpu_partition
    if overrides:
        cfg = dataclasses.replace(cfg, **overrides)
    if not cfg.partition:
        raise ValueError(
            f"stage {stage_name!r}: no partition resolved "
            f"(needs --gpu-partition for GPU stages, --cpu-partition for CPU)"
        )
    return cfg


def chunk(items: Sequence, size: int) -> list[list]:
    """Split items into chunks of at most `size`."""
    return [list(items[i:i + size]) for i in range(0, len(items), size)]


def submit_stage(
    stage_name: str,
    items: Sequence,
    fn: Callable,
    *,
    cpu_partition: str,
    gpu_partition: str,
    chunk_size: int,
    logs_dir: Path,
    overrides: dict | None = None,
):
    """Submit a stage via submitit AutoExecutor.

    Returns the list of submitit Job objects (one per chunk).
    Caller is responsible for waiting + collecting results (see
    progress.submitit_progress + job.results()).
    """
    import submitit  # imported lazily so paperprep is importable without slurm

    cfg = get_stage_config(
        stage_name,
        cpu_partition=cpu_partition,
        gpu_partition=gpu_partition,
        overrides=overrides,
    )
    chunks = chunk(items, chunk_size)
    folder = Path(logs_dir) / stage_name
    folder.mkdir(parents=True, exist_ok=True)

    executor = submitit.AutoExecutor(folder=str(folder))
    params = dict(
        slurm_partition=cfg.partition,
        timeout_min=cfg.timeout_min,
        cpus_per_task=cfg.cpus_per_task,
        mem_gb=cfg.mem_gb,
        slurm_array_parallelism=cfg.array_parallelism,
    )
    if cfg.gpus_per_node > 0:
        params["gpus_per_node"] = cfg.gpus_per_node
    params.update(cfg.extra_slurm)
    executor.update_parameters(**params)

    return executor.map_array(fn, chunks)
