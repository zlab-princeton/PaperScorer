"""SQLite-backed p_accept cache keyed by (reranker_ckpt_id, arxiv_id).

Why: vision-reranker inference is ~1-3s/paper; 200 candidates per query is
~7 min of GPU time uncached. Papers recur across queries on the same fixed
corpus, so a read-through cache turns most queries into retriever + lookup.

Bootstrap-friendly: ``bootstrap_from_parquet`` lets the operator pre-load
predictions produced by an offline batch (e.g. RANKER.md §6.1's
litsearch_eval/passover/predictions_3b.parquet). Bootstrap rows are tagged
``source='bootstrap'`` so they can be audited or invalidated separately
from rows produced by the running server.

WAL mode + a single shared connection is fine for the load profile (one
process, modest QPS, batched reads/writes).
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import logging
import sqlite3
from pathlib import Path
from typing import Iterable, Sequence


log = logging.getLogger(__name__)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS p_accept (
    reranker_ckpt_id TEXT NOT NULL,
    arxiv_id         TEXT NOT NULL,
    p_accept         REAL NOT NULL,
    computed_at      TEXT NOT NULL,
    source           TEXT NOT NULL DEFAULT 'online',
    compute_arch     TEXT NOT NULL DEFAULT 'unknown',
    PRIMARY KEY (reranker_ckpt_id, arxiv_id)
);
CREATE INDEX IF NOT EXISTS idx_arxiv_id ON p_accept(arxiv_id);
"""

# NOTE on cross-architecture drift:
# vLLM picks attention kernels per-GPU-class (e.g. H100/SXM vs gpu80 partition
# nodes). For the same model + same inputs + same vLLM/transformers version,
# bf16 logits drift on the order of:
#     mean   |Δp_accept| ~ 0.02
#     median |Δp_accept| ~ 0.01
#     max    |Δp_accept| ~ 0.21   (concentrated at p≈0.5)
#     binary Accept/Reject flip rate ~ 1.9%
#     Pearson r(scores_archA, scores_archB) ~ 0.994
# This means top-K reranking is preserved across architectures (recall@K
# essentially unchanged) but absolute p_accept values can differ. Tag every
# cache row with compute_arch so we can audit and, if needed, invalidate
# rows produced on a different architecture than the running deployment.


def hash_ckpt_id(ckpt_path: str | Path) -> str:
    """Short stable id derived from the ckpt path.

    We use a 12-hex-char SHA1 prefix — collisions are vanishingly unlikely
    across the handful of ckpts a single deployment will see, and the short
    form is easy to log.
    """
    s = str(Path(ckpt_path).resolve())
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


class PAcceptCache:
    """Read-through SQLite cache for p_accept scores."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute("PRAGMA synchronous = NORMAL")
        self.conn.executescript(_SCHEMA)
        self.conn.commit()
        log.info(f"PAcceptCache opened at {self.db_path} (rows={self.size()})")

    def close(self) -> None:
        self.conn.close()

    def size(self, ckpt_id: str | None = None) -> int:
        cur = self.conn.cursor()
        if ckpt_id is None:
            cur.execute("SELECT COUNT(*) FROM p_accept")
        else:
            cur.execute("SELECT COUNT(*) FROM p_accept WHERE reranker_ckpt_id = ?", (ckpt_id,))
        return int(cur.fetchone()[0])

    def get(self, ckpt_id: str, arxiv_ids: Sequence[str]) -> dict[str, float]:
        """Batch lookup. Returns a dict of {arxiv_id: p_accept} for hits only.

        Caller compares ``set(arxiv_ids) - set(result)`` to discover misses.
        """
        if not arxiv_ids:
            return {}
        # SQLite has a 999-parameter limit on default builds; chunk just in case.
        out: dict[str, float] = {}
        chunk = 500
        for i in range(0, len(arxiv_ids), chunk):
            batch = arxiv_ids[i: i + chunk]
            placeholders = ",".join("?" * len(batch))
            sql = (
                "SELECT arxiv_id, p_accept FROM p_accept "
                f"WHERE reranker_ckpt_id = ? AND arxiv_id IN ({placeholders})"
            )
            cur = self.conn.cursor()
            cur.execute(sql, [ckpt_id, *batch])
            for aid, p in cur.fetchall():
                out[aid] = float(p)
        return out

    def put(
        self,
        ckpt_id: str,
        scores: dict[str, float],
        *,
        source: str = "online",
        compute_arch: str = "unknown",
    ) -> int:
        """Atomic batched upsert. Returns the number of rows written.

        ``compute_arch`` should identify the hardware class that produced the
        scores (e.g. 'pli_h100_sxm', 'gputest_gpu80'). Cross-arch scores drift
        ~0.02 mean / 0.21 max for the same model + inputs -- see the schema
        comment for full numbers.
        """
        if not scores:
            return 0
        now = _dt.datetime.utcnow().isoformat(timespec="seconds")
        rows = [(ckpt_id, aid, float(p), now, source, compute_arch) for aid, p in scores.items()]
        cur = self.conn.cursor()
        cur.executemany(
            "INSERT OR REPLACE INTO p_accept "
            "(reranker_ckpt_id, arxiv_id, p_accept, computed_at, source, compute_arch) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
        self.conn.commit()
        return len(rows)

    def bootstrap_from_parquet(
        self,
        parquet_path: str | Path,
        *,
        ckpt_id: str,
        arxiv_id_col: str = "arxiv_id",
        p_accept_col: str | None = None,
        source: str = "bootstrap",
        compute_arch: str = "pli_h100_sxm",
    ) -> int:
        """One-time pre-load from a Parquet file.

        Auto-detects the p_accept column when ``p_accept_col`` is None: prefers
        ``p_accept_3b`` / ``p_accept_7b`` / ``p_accept``. Skips rows where
        ``(ckpt_id, arxiv_id)`` already exists (so re-running is a no-op).

        Returns the number of rows actually inserted.
        """
        try:
            import pandas as pd
        except ImportError as e:
            raise RuntimeError("bootstrap_from_parquet needs pandas + pyarrow installed") from e

        df = pd.read_parquet(parquet_path)
        if p_accept_col is None:
            for c in ("p_accept", "p_accept_3b", "p_accept_7b"):
                if c in df.columns:
                    p_accept_col = c
                    break
        if p_accept_col is None or p_accept_col not in df.columns:
            raise ValueError(
                f"could not find a p_accept column in {parquet_path} "
                f"(have {list(df.columns)})"
            )
        log.info(
            f"bootstrap_from_parquet: {len(df)} rows from {parquet_path} "
            f"col={p_accept_col} -> ckpt_id={ckpt_id}"
        )

        # Filter out (ckpt_id, arxiv_id) pairs we already have so re-runs are
        # cheap and don't overwrite online-computed scores.
        existing = set(self.get(ckpt_id, df[arxiv_id_col].tolist()).keys())
        scores = {
            row[arxiv_id_col]: float(row[p_accept_col])
            for _, row in df.iterrows()
            if row[arxiv_id_col] not in existing
        }
        self.put(ckpt_id, scores, source=source, compute_arch=compute_arch)
        return len(scores)
