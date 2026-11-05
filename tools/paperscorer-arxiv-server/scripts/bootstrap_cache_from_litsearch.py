#!/usr/bin/env python3
"""One-shot pre-load of the SQLite p_accept cache from a predictions parquet.

The offline LitSearch run produced exact p_accept values for ~28,664 papers in
the per_venue candidate pool, byte-identical to what online vLLM would compute
(greedy decode + 2-token softmax at the boxed slot). Dropping them into the
SQLite cache skips that GPU work on first query. The parquets ride along in
the HF index bundle (``skonan/PaperScorer-arXiv-embeddings`` -> ``cache/``);
``paperlens serve --with-retrieval`` runs this script automatically once the
scorer is up.

Run it ONCE per deployment, after the scorer is serving (the ckpt_path comes
from its /health). Safe to re-run: existing (ckpt_id, arxiv_id) rows are
preserved (online-computed scores never overwritten).

Example:

    python scripts/bootstrap_cache_from_litsearch.py \\
        --parquet <hf_cache>/datasets--skonan--PaperLens-arXiv-embeddings/.../cache/predictions_3b.parquet \\
        --ckpt_path <ckpt_path the scorer reports at /health> \\
        --db_path ./cache/p_accept.sqlite

The --ckpt_path is hashed the same way the server hashes it, so the
bootstrapped rows are visible to the running server when it's pointed at
the same ckpt.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make the package importable when this script is run from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from paperlens_arxiv_server.cache import PAcceptCache, hash_ckpt_id  # noqa: E402


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet", required=True, help="Path to predictions_*.parquet")
    ap.add_argument("--ckpt_path", required=True, help="Reranker ckpt_path (same as configs/server.yaml)")
    ap.add_argument("--db_path", default="./cache/p_accept.sqlite", help="SQLite cache DB path")
    ap.add_argument("--p_accept_col", default=None,
                    help="Column name in the parquet. Auto-detects p_accept / p_accept_3b / p_accept_7b.")
    ap.add_argument("--arxiv_id_col", default="arxiv_id")
    ap.add_argument("--source", default="bootstrap")
    args = ap.parse_args()

    ckpt_id = hash_ckpt_id(args.ckpt_path)
    print(f"ckpt_id (sha1[:12]): {ckpt_id}")
    cache = PAcceptCache(args.db_path)
    before = cache.size(ckpt_id)
    inserted = cache.bootstrap_from_parquet(
        args.parquet,
        ckpt_id=ckpt_id,
        arxiv_id_col=args.arxiv_id_col,
        p_accept_col=args.p_accept_col,
        source=args.source,
    )
    after = cache.size(ckpt_id)
    print(f"rows for ckpt_id before: {before}")
    print(f"rows inserted from parquet: {inserted}")
    print(f"rows for ckpt_id after:  {after}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
