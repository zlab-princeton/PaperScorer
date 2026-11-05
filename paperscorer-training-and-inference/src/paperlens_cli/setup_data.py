"""``paperlens setup-data`` — materialize sharegpt datasets from the HF release.

Thin wrapper around ``scripts/reconstruction.py``: dispatches the CLI args
through to that script's ``main``, then prints what was written.

The reconstruction is byte-identical to the source sharegpt the model was
trained on (validated by ``scripts/tests/test_reconstruction.py``), so
materializing here means the downstream ``paperlens serve`` consumer gets
the same token ids it would see in training.
"""
from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path


def _build_reconstruction_argv(args: argparse.Namespace) -> list[str]:
    """Translate `paperlens setup-data ...` args into reconstruction.py CLI argv."""
    argv: list[str] = ["scripts/reconstruction.py"]
    if args.all:
        argv.append("--all")
    if args.dataset_keys:
        argv += ["--dataset_keys", *args.dataset_keys]
    if args.local_dir:
        argv += ["--local_dir", args.local_dir]
    if args.hf_text_repo:
        argv += ["--hf_text_repo", args.hf_text_repo]
    if args.hf_vision_repo:
        argv += ["--hf_vision_repo", args.hf_vision_repo]
    if args.data_root:
        argv += ["--data_root", args.data_root]
    return argv


def run(args: argparse.Namespace) -> int:
    # Import reconstruction.py from scripts/. We re-import each call so a
    # second invocation in the same process picks up edits during dev.
    repo_root = Path(__file__).resolve().parents[2]
    scripts_dir = repo_root / "scripts"
    if not (scripts_dir / "reconstruction.py").exists():
        print(
            f"ERROR: {scripts_dir / 'reconstruction.py'} not found "
            "(expected to be sibling of paperlens_cli/)",
            file=sys.stderr,
        )
        return 2

    argv = _build_reconstruction_argv(args)
    print(f"[setup-data] invoking: python {shlex.join(argv)}")

    # Push scripts/ onto sys.path, then call reconstruction.main() with sys.argv
    # swapped so its argparse reads our composed argv. (Simpler than refactoring
    # reconstruction.py to accept argv as a parameter.)
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import reconstruction  # type: ignore  # noqa: E402

    saved_argv = sys.argv
    try:
        sys.argv = argv          # reconstruction.py's argparse reads sys.argv
        return reconstruction.main()
    finally:
        sys.argv = saved_argv
