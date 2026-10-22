# arxiv_retriever

Standalone FAISS-based arXiv retrieval server with no-lookahead-bias filtering and a DeepResearch agent integration.

Structure:

```
src/arxiv_retriever/
├── server/                    # FastAPI /retrieve endpoint with upper_bound_datetime + exclude_title
│   ├── retrieval_server.py    # OmegaConf-driven main
│   ├── retrieval_launch.sh    # background launcher (expects conda `retriever` env active)
│   ├── index_builder.py
│   └── build_index.sh
├── clients/
│   └── arxiv_client.py        # plain-Python client for /retrieve
└── integrations/
    └── deepresearch/          # qwen-agent tool + runner for Tongyi-DeepResearch-30B-A3B
        ├── tool_arxiv_search.py
        └── dr_agent_runner.py

configs/retrieval/
├── qwen3_4b.yaml              # default: Qwen3-Embedding-4B
└── qwen3_06b.yaml             # fallback: Qwen3-Embedding-0.6B
```

## Two environments — server vs. agent

The retrieval server needs `faiss-gpu`, which is conda-only, so it lives in a dedicated **conda** env (`retriever`). Any client of the server (e.g. the DeepResearch agent runner) talks to it over HTTP on port 8000 and can run in whatever Python env is convenient — typically a **uv venv** installed from this repo's `[deepresearch]` extra.

### Server env (conda)

Follow the SkyRL search retriever recipe (https://github.com/NovaSky-AI/SkyRL/blob/main/examples/train/search/README.md), or use the bundled `conda-env.yml`:

```bash
module load anaconda3/2024.6
conda env create -f conda-env.yml           # first time (name: retriever)
# or if you already have the env from SkyRL:
conda activate retriever
pip install omegaconf pandas
pip install -e /scratch/gpfs/ZHUANGL/sk7524/arxiv_retriever
```

### Agent env (uv)

For anything that uses the `ArxivClient` or the DeepResearch integration:

```bash
cd /scratch/gpfs/ZHUANGL/sk7524/arxiv_retriever
uv venv
source .venv/bin/activate
uv pip install -e '.[deepresearch]'
```

The DR runner only imports `arxiv_retriever.clients.arxiv_client` (a plain-requests HTTP client) and `arxiv_retriever.integrations.deepresearch.*` (qwen-agent glue). It never touches FAISS, so the uv env stays clean.

## Launch the server

```bash
module load anaconda3/2024.6
conda activate retriever
bash src/arxiv_retriever/server/retrieval_launch.sh configs/retrieval/qwen3_4b.yaml

# Or with dotted OmegaConf overrides:
bash src/arxiv_retriever/server/retrieval_launch.sh configs/retrieval/qwen3_4b.yaml server.port=8001

# Test:
curl -X POST http://localhost:8000/retrieve \
  -H 'Content-Type: application/json' \
  -d '{"query": "sparse attention transformer", "topk": 3, "upper_bound_datetime": "2022-01-01"}'
```

## No-lookahead enforcement

The server accepts `upper_bound_datetime` (ISO date) and `exclude_title` on every request. Any paper published after the cutoff, or whose title fuzzy-matches `exclude_title` (≥98% similarity), is dropped server-side before top-k is returned.

The DeepResearch integration (`integrations/deepresearch/tool_arxiv_search.py`) wraps this as a qwen-agent tool and **refuses to call the server** unless `set_task_config(upper_bound_datetime, exclude_title)` has been called on the current thread. This is a hard guarantee against silent lookahead bias.
