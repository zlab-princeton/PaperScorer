# Agent / LLM Guide: paperscorer-arxiv-server

Entry point for any agent (Claude Code, Codex, Cursor, etc.) landing in
the repo. `README.md` is for humans installing + running the service;
this file is for agents navigating the code. `CLAUDE.md` is a symlink to
this file — both CLIs pick up the same content from the cwd.

---

## What this repo is

A **search/rerank orchestration service**: a retriever (FAISS or BM25)
finds candidates from the ~80K per_venue arXiv subset, then a PaperScorer
vision model reranks them by predicted-acceptance probability. Exposes a search UI
(`GET /`) and JSON endpoints (`/search`, `/search_compare`, `/health`).

This server is **orchestration only** — no GPU, no model weights. Two
upstreams do the work:

- `external/arxiv_retriever` (conda env `retriever`) —
  FAISS / BM25 server, port 8001.
- `paperlens-serve` (in `../../paperscorer-training-and-inference`) — vLLM
  scorer; we POST sharegpt rows to its `/score` endpoint.

Single human doc: [`README.md`](README.md) — quickstart, endpoints, UI,
config, drift note.

---

## Where to start a task

| Task | Start here |
|---|---|
| Local dev launch | [`README.md`](README.md) → Quickstart. `bash scripts/launch_local.sh`. |
| Provision through the orchestrator | `paperscorer setup --with-retrieval` then `paperscorer serve --with-retrieval` (repo root `../..`). Code lives in `../../src/paperscorer_orchestrator/retrieval.py` + `workflow.serve_retrieval`. |
| Change rerank pool size / blend weights | `configs/server.yaml` — `search.rerank_pool`, `search.compare_top`, `search.blend_weight_*`. |
| Add a new endpoint | `src/paperlens_arxiv_server/server.py` — define pydantic models, add a `@app.post(...)` handler. Reuse `_rank_pool` if it needs retrieval + rerank. |
| Modify the search UI | `src/paperlens_arxiv_server/ui/index.html` — single-file vanilla JS, no build step. Served by `GET /`. |
| Debug "no cache hits" | `cache.py`. Rows are keyed by `(ckpt_id, arxiv_id)` where `ckpt_id = sha1[:12](scorer's ckpt_path)`. Confirm via `GET /health` `ckpt_id` matches what was bootstrapped. |
| Seed the cache | `scripts/bootstrap_cache_from_litsearch.py`. Auto-runs from `paperscorer serve --with-retrieval` once the scorer reports its ckpt_path. |
| Upload a new index bundle to HF | `scripts/upload_indexes_to_hf.py`. |
| Change retriever (qwen ↔ bm25) | `cfg.retrieval.retriever` in the orchestrator. The orchestrator's `render_retriever_config` emits the matching yaml (`name: qwen3` / `name: bm25`). |
| Investigate a wrong arxiv_id in results | `external/arxiv_retriever` `retrieval_server.py` resolves faiss-pos → arxiv_id via `corpus[idx]['id']` then `arxiv_id_to_pos`. Subset-index bug: see `eca2e65` in arxiv_retriever. |

---

## Hard rules

1. **No model weights, no GPU here.** The reranker is an HTTP client to
   `paperlens-serve` (`reranker.py`); the retriever is an HTTP client to
   `arxiv_retriever` (`retriever_client.py`). Don't pull torch / vLLM
   into this server — it should boot on a CPU-only laptop.
2. **Cache key = scorer's runtime ckpt_path.** At startup we probe
   `paperlens-serve /health`, hash the reported `ckpt_path`, and use
   that as `ckpt_id`. The bootstrap step must use the **same** ckpt_path
   or its rows are invisible to the running server.
3. **Path layout must match `paperscorer_orchestrator.retrieval.BUNDLE`.**
   `scripts/upload_indexes_to_hf.py::DEST` and the orchestrator's
   `BUNDLE` dict are two halves of the same contract — change either,
   change both.
4. **`/search` and `/search_compare` share `_rank_pool`.** Don't
   duplicate the retrieve → rerank → blend logic in a new endpoint.
5. **No hardcoded `/scratch/...` paths.** All defaults read from
   `configs/server.yaml` or env (`PAPERLENS_DATA_ROOT`,
   `PAPERLENS_IMAGES_ROOT`, `HF_HOME`). The upload script's
   `--bm25 / --faiss / --corpus` are REQUIRED args (no cluster-specific defaults).
6. **`compute_arch` is auditable.** Every cache row is tagged with the
   GPU class that produced it. Mixed-arch caches are operationally fine
   but accountable — `SELECT compute_arch, COUNT(*) FROM p_accept GROUP BY 1`.

---

## Critical invariants downstream relies on

- **`RankedPaper.{rerank_position, retriever_position}` are global**
  within the reranked pool, not within the returned slice. The UI uses
  this to render movement badges (`▲N` / `▼N` between the two columns).
  Don't recompute them per-slice.
- **The rerank pool is fully reranked.** Reranked top-20 can pull from
  anywhere in the pool — that's the demonstrative value over base. If
  you cap the rerank pass to top-20-only, the side-by-side becomes a
  reordering instead of a re-selection.
- **Bootstrap completes before the arxiv-server starts.** Sequence:
  scorer up → bootstrap runs once → arxiv-server execs. Cache must be
  warm at request time — concurrent bootstrap + serve races on SQLite.
- **`images_root` is the per-paper-dir root** (`<root>/<arxiv_id>/page_N.png`),
  not a flat dump. Resolver order: `cfg.images_root` → `$PAPERLENS_IMAGES_ROOT`
  → `$PAPERLENS_DATA_ROOT/images_arxiv` → error.
- **`start_awake: true` in the qwen retriever config** loads the embed
  model on GPU at boot. Without it, the model parks on CPU and query
  embedding is slow. The orchestrator emits this; don't drop it.

---

## Sibling directories

- [`external/arxiv_retriever`](external/arxiv_retriever) — separate
  **conda** env (faiss-gpu). The actual FAISS / BM25
  server lives there. We POST `/retrieve`. Per-venue subset-index fix
  (`arxiv_id_to_pos`) landed at commit `eca2e65`; older pins return
  wrong arxiv_ids on subset indexes (0% recall).
- [`../../paperscorer-training-and-inference`](../../paperscorer-training-and-inference)
  — provides `paperlens-serve` (vLLM, the reranker checkpoint). Its
  `/health` reports `ckpt_path` which we hash for the cache key.
- [`../..`](../..) — the PaperScorer repo root (orchestrator). `paperscorer setup
  --with-retrieval` and `paperscorer serve --with-retrieval` drive the
  whole stack; the relevant code is `src/paperscorer_orchestrator/retrieval.py`
  and `workflow.serve_retrieval`.

If you change the bundle layout, the SQLite cache schema, or the
`RankedPaper` shape, grep the orchestrator + `external/arxiv_retriever`
(the `arxiv_id_to_pos` resolution path) before merging.

---

## Maintainer-only

```bash
# unit tests (no GPU)
uv run pytest tests/test_cache.py -v

# reranker smoke — points at a running paperlens-serve
PAPERLENS_SERVE_URL=http://localhost:8002 \
    uv run pytest tests/test_reranker.py -v

# end-to-end against a running `bash scripts/launch_local.sh` stack
PAPERLENS_E2E_URL=http://localhost:8000 \
    uv run pytest tests/test_e2e.py -v
```

Upload a fresh index bundle (token via `huggingface-cli login` or `$HF_TOKEN`):

```bash
python scripts/upload_indexes_to_hf.py \
    --bm25 /path/to/bm25_per_venue_index/ \
    --faiss /path/to/qwen3_Flat.index \
    --corpus /path/to/arxiv_wikiformat_per_venue.jsonl \
    [--pred3b /path/to/predictions_3b.parquet] \
    [--pred7b /path/to/predictions_7b.parquet] \
    [--dry-run]
```
