# paperlens-arxiv-server

A search/rerank service for arXiv papers. A retriever (FAISS or BM25) finds
candidates from the ~80K **per_venue** subset; **PaperLens-V** then reranks
them by predicted-acceptance probability. Exposes a query UI and a JSON API.

```
user → POST /search_compare {query}
  paperlens-arxiv-server  (this repo, port 8000)
    1. retriever.retrieve(query, topk=pool)        → candidates by retriever rank
    2. cache.get(ckpt_id, arxiv_ids)               → partition into hits / misses
    3. reranker.score(misses)                      → POST /score to paperlens-serve (vLLM)
    4. cache.put(misses)                           → online + bootstrap rows persist
    5. blend = 0.3·z(retriever_score) + 0.7·p_accept
  → { base: [retriever order, top 20],
      reranked: [blend order,    top 20] }
```

Two upstream services do the heavy lifting:

- **arxiv_retriever** (`external/arxiv_retriever`, conda env `retriever`) — serves the
  FAISS / BM25 index over the per_venue corpus.
- **paperlens-serve** (from `paperlens-training-and-inference`) — vLLM-hosts the
  reranker checkpoint (`PaperLens-V-3B` by default) and exposes `/score`.

This server is **orchestration only** — no GPU, no model weights. The
recommended way to provision and launch the whole stack is via the
top-level `paperlens` CLI:

```bash
paperlens setup --with-retrieval     # provision data + indexes + conda env + config
paperlens serve --with-retrieval     # launch scorer + retriever + UI; opens the search UI
```

This README documents the standalone setup (without the orchestrator) and the
HTTP API.

---

## Quickstart

### 1. Environments

The retriever is **conda** (faiss-gpu is conda-only); this server is **uv/pip**.

```bash
# Run from the paperlens-arxiv-server directory: it's already present
# inside the PaperLens release (`cd PaperLens/tools/paperlens-arxiv-server`),
# or clone the standalone repo with `--recurse-submodules` to pick up
# external/arxiv_retriever.

# arxiv_retriever conda env (one-off; needs faiss-gpu + pyserini + Java 21)
conda env create -f external/arxiv_retriever/conda-env.yml
# -> creates env `retriever`. Land it next to your other envs with:
#    CONDA_ENVS_PATH=/path/to/.conda conda env create -f external/arxiv_retriever/conda-env.yml

# this server (uv)
uv sync                              # creates .venv, installs the package + deps
```

### 2. Indexes + p_accept cache seed (from Hugging Face)

`paperlens setup --with-retrieval` clones what it needs from the
[`skonan/PaperScorer-arXiv-embeddings`](https://huggingface.co/datasets/skonan/PaperScorer-arXiv-embeddings)
dataset into your HF cache:

| path in repo | what |
|---|---|
| `bm25_per_venue_index/` | pyserini/lucene BM25 index over the per_venue corpus |
| `qwen3_06b_per_venue/qwen3_Flat.index` | Qwen3-Embedding-0.6B dense FAISS (Flat) index |
| `corpus/arxiv_wikiformat_per_venue.jsonl` | shared corpus (resolves FAISS position → arxiv_id) |
| `cache/predictions_{3b,7b}.parquet` | precomputed p_accept seed (~28K papers; loaded into the SQLite cache at serve time) |

If you're not using the orchestrator, point the retriever config at the cloned
paths yourself (see `external/arxiv_retriever/configs/retrieval/qwen3_06b_per_venue.yaml`
as a template) or use the standalone defaults in `configs/server.yaml`.

You also need the reranker's **per-paper page PNGs** somewhere on disk
(`<images_root>/<arxiv_id>/page_N.png`). Point `images_root` in
`configs/server.yaml` at that directory (or set `$PAPERLENS_IMAGES_ROOT`).

### 3. Launch

```bash
bash scripts/launch_local.sh         # starts arxiv_retriever (:8001) + this server (:8000)
# logs land in ./logs/
```

`launch_local.sh` reads `RETRIEVER_CFG` (default: the per_venue qwen3 config; the
orchestrator writes a fully-resolved `configs/retriever.generated.yaml` when you
run `paperlens setup --with-retrieval`).

Open the search UI at `http://<host>:8000/`.

---

## Endpoints + UI

### `GET /` — search UI

A minimal page: query box → two columns of results.

- **Base** — top 20 in retriever order.
- **Reranked** — top 20 after the PaperLens blend.

Each result shows `p_accept`, retriever score, and **movement** between the two
orderings (▲N up / ▼N down from the corresponding rank in the other column),
making it easy to see which papers the reranker pulled up or pushed down.

### `POST /search_compare` — side-by-side JSON

Powers the UI. Retrieves and reranks a pool (default 50 — see
`search.rerank_pool`), returns the top 20 in each ordering.

```bash
curl -X POST http://localhost:8000/search_compare \
     -H 'content-type: application/json' \
     -d '{"query": "sparse attention for long context"}'
```

Response shape:

```jsonc
{
  "query": "...",
  "pool": 50,
  "n_cache_hits": 36,        // of the pool, served from the SQLite cache
  "n_inferred": 14,          // of the pool, freshly scored by paperlens-serve
  "base":     [/* RankedPaper × 20, retriever order */],
  "reranked": [/* RankedPaper × 20, blend order */]
}
```

Each `RankedPaper`:

```jsonc
{
  "paper_id": "2311.04567", "title": "...", "abstract": "...",
  "p_accept": 0.82,           // softmax([logp_accept, logp_reject]) in [0,1]
  "retriever_score": 0.71,    // raw upstream similarity
  "blend_score": 0.78,        // 0.3·z(retriever_score) + 0.7·p_accept
  "rerank_position": 3,       // global rank within the pool by blend
  "retriever_position": 11,   // global rank within the pool by retriever
  "cache_hit": true,
  "submission_date": null
}
```

The two position fields make per-paper movement renderable client-side without
a second request.

### `POST /search` — single-list JSON (legacy API)

Returns the top-`k` reranked papers as one list. Same `RankedPaper` shape; `k`
and `rerank_pool` are tunable per request (see `SearchRequest` in `server.py`).

### `GET /health`

Service info + upstream status:

```jsonc
{
  "status": "ok",
  "paperlens_serve_url": "...",
  "ckpt_id": "777c771532ac",          // sha1[:12] of the reranker ckpt_path (cache key)
  "compute_arch": "h100_sxm",         // tagged on each new cache row
  "retriever_url": "...",
  "images_root": "...",
  "cache_db": "./cache/p_accept.sqlite",
  "cache_rows_for_ckpt": 28664
}
```

---

## Configuration (`configs/server.yaml`)

| key | what |
|---|---|
| `server.{host,port}` | this service's bind address |
| `retriever.{base_url, topk_retrieve, timeout_seconds}` | arxiv_retriever endpoint + default pool size |
| `paperlens_serve.{base_url, timeout_seconds}` | paperlens-serve `/score` endpoint |
| `reranker.{domain, modality}` | `arxiv|iclr` × `text|vision` (drives the prompt + sharegpt row composition) |
| `images_root` | per-paper page-image root; falls back to `$PAPERLENS_IMAGES_ROOT` then `$PAPERLENS_DATA_ROOT/images_arxiv` |
| `cache.db_path` | SQLite cache location (read-through; persists across restarts) |
| `search.{default_blend, blend_weight_retriever, blend_weight_p_accept}` | blend mode + weights (defaults per RANKER.md §5.2) |
| `search.{rerank_pool, compare_top}` | `/search_compare` pool size and per-column row count |

---

## Tests

```bash
uv run pytest tests/test_cache.py -v          # cache unit tests (no GPU)
uv run pytest tests/test_reranker.py -v       # reranker smoke (vLLM; needs a ckpt)
PAPERLENS_E2E_URL=http://localhost:8000 \
    uv run pytest tests/test_e2e.py -v        # against a running launch_local.sh stack
```

---

## Cross-architecture p_accept drift (worth knowing)

vLLM picks attention kernels per-GPU-class, so the same model + same inputs +
same vLLM/transformers version can produce bf16 logits that drift across
hardware. For the 3B vision reranker:

| stat | value (H100-SXM vs gpu80, 1000-row diag) |
|---|---|
| Pearson r(p_A, p_B) | **0.994** |
| Mean &#124;Δp_accept&#124; | 0.022 |
| Max &#124;Δp_accept&#124; | 0.214 (concentrated at p≈0.5) |
| Binary Accept/Reject flips | 1.9 % |

**Ranks are preserved** (Pearson > 0.99) so recall@K stays valid; absolute
p_accept can differ by up to ~0.2 on borderline papers. Every cache row is
tagged with the originating `compute_arch`, so mixed-arch caches stay
auditable: `SELECT compute_arch, COUNT(*) FROM p_accept GROUP BY 1`.
