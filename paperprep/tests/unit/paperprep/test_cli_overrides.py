"""Tests for paperprep.cli — argparse + override parsing."""
import pytest

from paper_anonymizer.paperprep.cli import build_parser, parse_overrides


def test_parse_overrides_int_and_str():
    out = parse_overrides("mineru.timeout_min=480,compile.mem_gb=16,export.partition=quick")
    assert out == {
        "mineru": {"timeout_min": 480},
        "compile": {"mem_gb": 16},
        "export": {"partition": "quick"},
    }


def test_parse_overrides_empty():
    assert parse_overrides("") == {}
    assert parse_overrides(None) == {}


def test_parse_overrides_bad_format_raises():
    with pytest.raises(ValueError):
        parse_overrides("not_a_valid_spec")


def test_parser_supports_run_subcommand():
    p = build_parser()
    args = p.parse_args([
        "run",
        "--input-manifest", "x.jsonl",
        "--output-dir", "/tmp/out",
        "--gpu-partition", "gpu",
        "--cpu-partition", "all",
    ])
    assert args.cmd == "run"
    assert args.gpu_partition == "gpu"
    assert args.chunk_size == 50
    assert args.resume is True
