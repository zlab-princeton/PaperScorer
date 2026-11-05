"""Upload the per_venue retrieval index bundle + p_accept cache seed to HF.

Serve-minimal bundle the deployment downloads at setup/serve time
(`paperlens setup --with-retrieval` clones from this repo, then the serve flow
seeds the SQLite cache from the predictions parquets):

    <repo>/
      bm25_per_venue_index/                     lucene BM25 index (dir)        [required]
      qwen3_06b_per_venue/qwen3_Flat.index      Qwen3-Embedding-0.6B FAISS     [required]
      corpus/arxiv_wikiformat_per_venue.jsonl   per_venue corpus               [required]
      cache/predictions_3b.parquet              precomputed p_accept, 3B ckpt  [optional]
      cache/predictions_7b.parquet              precomputed p_accept, 7B ckpt  [optional]

The path layout MUST match the orchestrator's ``BUNDLE`` dict (see
``paperlens_orchestrator/retrieval.py``).

Idempotent: re-running uploads only changed files. Needs an HF token
(``huggingface-cli login`` or ``$HF_TOKEN``).

Usage:

    python scripts/upload_indexes_to_hf.py \\
        --bm25   /path/to/arxiv_bm25_per_venue_index/ \\
        --faiss  /path/to/qwen3_Flat.index \\
        --corpus /path/to/arxiv_wikiformat_per_venue.jsonl \\
        [--pred3b /path/to/predictions_3b.parquet] \\
        [--pred7b /path/to/predictions_7b.parquet] \\
        [--repo  skonan/PaperScorer-arXiv-embeddings] \\
        [--private] [--dry-run]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

DEFAULT_REPO = "skonan/PaperScorer-arXiv-embeddings"

# Bundle layout (path_in_repo) -- keep in sync with orchestrator retrieval.BUNDLE.
DEST = {
    "bm25":   "bm25_per_venue_index",                  # folder
    "faiss":  "qwen3_06b_per_venue/qwen3_Flat.index",  # file
    "corpus": "corpus/arxiv_wikiformat_per_venue.jsonl",  # file
    "pred3b": "cache/predictions_3b.parquet",          # file (optional)
    "pred7b": "cache/predictions_7b.parquet",          # file (optional)
}

REQUIRED = ("bm25", "faiss", "corpus")    # serve can't run without these
OPTIONAL = ("pred3b", "pred7b")           # cache seed -- speeds up first queries

README = """\
---
license: cc-by-4.0
tags: [paperlens, retrieval, arxiv, bm25, faiss, qwen3-embedding]
---

# PaperLens arXiv per_venue retrieval indexes

Index bundle + p_accept cache seed for the PaperLens arxiv search/rerank
deployment (`paperlens-arxiv-server` + `arxiv_retriever`). Covers the ~80K
per_venue arXiv corpus.

| path | what |
|---|---|
| `bm25_per_venue_index/` | Lucene BM25 index (pyserini) over the per_venue corpus |
| `qwen3_06b_per_venue/qwen3_Flat.index` | Qwen3-Embedding-0.6B dense FAISS (Flat) index |
| `corpus/arxiv_wikiformat_per_venue.jsonl` | per_venue corpus (resolves FAISS position -> arxiv_id) |
| `cache/predictions_3b.parquet` | precomputed p_accept for the 3B reranker (~28,664 papers) |
| `cache/predictions_7b.parquet` | precomputed p_accept for the 7B reranker (~28,664 papers) |

`paperlens setup --with-retrieval` downloads this into the HF cache and renders
an `arxiv_retriever` config; `paperlens serve --with-retrieval` bootstraps the
SQLite p_accept cache from the matching predictions parquet (keyed by the
serving ckpt). The rebuild-only embedding memmap + full arxiv metadata
snapshot are intentionally excluded.
"""


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--repo", default=os.environ.get("PAPERLENS_INDEX_REPO", DEFAULT_REPO),
                    help=f"HF dataset repo id (default: {DEFAULT_REPO}; or $PAPERLENS_INDEX_REPO)")
    ap.add_argument("--bm25", type=Path, required=True,
                    help="path to the BM25 lucene index DIRECTORY")
    ap.add_argument("--faiss", type=Path, required=True,
                    help="path to the qwen3-0.6b FAISS .index file")
    ap.add_argument("--corpus", type=Path, required=True,
                    help="path to arxiv_wikiformat_per_venue.jsonl")
    ap.add_argument("--pred3b", type=Path, default=None,
                    help="optional: predictions_3b.parquet (seeds the p_accept cache)")
    ap.add_argument("--pred7b", type=Path, default=None,
                    help="optional: predictions_7b.parquet (seeds the p_accept cache)")
    ap.add_argument("--private", action="store_true", help="create the repo private")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src = {k: getattr(args, k) for k in DEST}
    missing_req = [f"{k}: {src[k]}" for k in REQUIRED if not src[k].exists()]
    if missing_req:
        print("ERROR: missing required source artifacts:", file=sys.stderr)
        for m in missing_req:
            print(f"  {m}", file=sys.stderr)
        return 2

    def _size(p: Path) -> str:
        n = (sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
             if p.is_dir() else p.stat().st_size)
        return f"{n / 1e6:.1f} MB"

    # which keys actually upload (required + any provided & existing optional)
    keys = list(REQUIRED) + [k for k in OPTIONAL if src[k] and src[k].exists()]
    skipped = [k for k in OPTIONAL if not src[k] or not src[k].exists()]

    print(f"repo: {args.repo}  (dataset, {'private' if args.private else 'public'})")
    for k in keys:
        kind = "dir " if src[k].is_dir() else "file"
        print(f"  {kind} {src[k]}  ->  {DEST[k]}   [{_size(src[k])}]")
    for k in skipped:
        print(f"  skip {k} (not provided or missing)")
    if args.dry_run:
        print("\n(dry-run — nothing uploaded)")
        return 0

    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset", private=args.private, exist_ok=True)
    api.upload_file(path_or_fileobj=README.encode(), path_in_repo="README.md",
                    repo_id=args.repo, repo_type="dataset")
    for k in keys:
        print(f"uploading {k} -> {DEST[k]} ...")
        if src[k].is_dir():
            api.upload_folder(folder_path=str(src[k]), path_in_repo=DEST[k],
                              repo_id=args.repo, repo_type="dataset")
        else:
            api.upload_file(path_or_fileobj=str(src[k]), path_in_repo=DEST[k],
                            repo_id=args.repo, repo_type="dataset")

    print(f"\n✓ uploaded to https://huggingface.co/datasets/{args.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
