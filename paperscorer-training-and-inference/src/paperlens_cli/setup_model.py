"""``paperlens setup-model`` — pull a PaperLens HF checkpoint to local cache.

Wraps ``huggingface_hub.snapshot_download``. Idempotent (HF cache de-dupes).
"""
from __future__ import annotations

import argparse
import sys
from typing import Optional


# The 8 published PaperLens repos (4 sizes × 2 modalities × {arxiv, iclr}).
KNOWN_REPOS: tuple[str, ...] = (
    "skonan/paperscorer-3b-text-arxiv",
    "skonan/paperscorer-7b-text-arxiv",
    "skonan/paperscorer-3b-vision-arxiv",
    "skonan/paperscorer-7b-vision-arxiv",
    "skonan/paperscorer-3b-text-openreview-iclr",
    "skonan/paperscorer-7b-text-openreview-iclr",
    "skonan/paperscorer-3b-vision-openreview-iclr",
    "skonan/paperscorer-7b-vision-openreview-iclr",
)


def _list_known() -> int:
    print("Known PaperLens HF repos:")
    for r in KNOWN_REPOS:
        print(f"  {r}")
    return 0


def _snapshot_download(hf_repo: str, revision: Optional[str], local_dir: Optional[str]) -> str:
    """Download (or refresh) a HF repo, return the resolved local path."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        print(f"ERROR: huggingface_hub not installed: {e}", file=sys.stderr)
        raise

    kwargs = {"repo_id": hf_repo}
    if revision:
        kwargs["revision"] = revision
    if local_dir:
        kwargs["local_dir"] = local_dir
    return snapshot_download(**kwargs)


def run(args: argparse.Namespace) -> int:
    if args.list:
        return _list_known()
    if not args.hf_repo:
        print("ERROR: --hf_repo is required (or use --list)", file=sys.stderr)
        return 2

    print(f"[setup-model] resolving {args.hf_repo!r} ...")
    path = _snapshot_download(args.hf_repo, args.revision, args.local_dir)
    print(f"[setup-model] local path: {path}")
    return 0
