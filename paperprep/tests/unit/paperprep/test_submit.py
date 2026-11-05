"""Tests for paperprep.submit — chunking + StageConfig resolution."""
import pytest

from paper_anonymizer.paperprep.submit import (
    DEFAULT_STAGE_CONFIGS,
    chunk,
    get_stage_config,
)


def test_chunk_splits_evenly():
    assert chunk(list(range(10)), 3) == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]


def test_chunk_size_larger_than_items():
    assert chunk([1, 2], 5) == [[1, 2]]


def test_chunk_empty():
    assert chunk([], 5) == []


def test_default_stage_configs_present():
    assert set(DEFAULT_STAGE_CONFIGS) == {"compile", "mineru", "normalize", "filter", "export"}


def test_mineru_uses_gpu_partition():
    cfg = get_stage_config("mineru", cpu_partition="cpu", gpu_partition="gpu")
    assert cfg.partition == "gpu"
    assert cfg.gpus_per_node == 1


def test_compile_uses_cpu_partition():
    cfg = get_stage_config("compile", cpu_partition="cpu", gpu_partition="gpu")
    assert cfg.partition == "cpu"
    assert cfg.gpus_per_node == 0


def test_overrides_are_applied():
    cfg = get_stage_config(
        "mineru", cpu_partition="cpu", gpu_partition="gpu",
        overrides={"timeout_min": 480, "mem_gb": 96},
    )
    assert cfg.timeout_min == 480
    assert cfg.mem_gb == 96
    assert cfg.gpus_per_node == 1  # not overridden


def test_unknown_stage_raises():
    with pytest.raises(KeyError):
        get_stage_config("nosuch", cpu_partition="cpu", gpu_partition="gpu")


def test_missing_partition_raises():
    with pytest.raises(ValueError):
        # No gpu_partition supplied; mineru needs one
        get_stage_config("mineru", cpu_partition="cpu", gpu_partition="")
