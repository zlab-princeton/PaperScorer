# paperlens-reviewing

A small FastAPI service + browser UI that **reviews one paper end-to-end**: drop
in a PDF, watch a progress bar walk through the pipeline, get back a verdict
(Accept / Reject) and the model's `p_accept` confidence.

It glues together the rest of the PaperLens ecosystem:

```
[ PDF upload ]
     │
     ▼  paperprep run (subprocess)
[ compile (skipped for PDF) → MinerU OCR → normalize → filter → export ShareGPT ]
     │
     ▼  POST /score
[ paperlens serve  (vLLM + LF tokenizer + qwen2_vl, p_accept softmax) ]
     │
     ▼
[ verdict + p_accept → browser UI ]
```

Sibling repos referenced:

- [`paperprep`](https://github.com/SachinKonan/AutoReviewer/tree/paperprep-only) — paper → ShareGPT pipeline
- [`paperlens-training-and-inference`](https://github.com/SachinKonan/LLaMA-Factory-AutoReviewer/tree/published-branch) — provides `paperlens serve`
- [`paperlens-arxiv-server`](https://github.com/SachinKonan/paperlens-arxiv-server) — different deployment (arxiv retriever stack)

---

## Layout

```
paperlens-reviewing/
├── pyproject.toml                       console_script: paperlensreview = …
├── README.md
├── scripts/launch_local.sh              GPU check → paperlens serve → paperlensreview → URL
├── configs/server.yaml                  ports, paperprep + paperlens locations, modality
├── src/paperlensreview/
│   ├── __main__.py                      CLI dispatch (subcommand: serve)
│   ├── server.py                        FastAPI: /submit, /status, /health, /
│   ├── pipeline.py                      PDF → paperprep run → paperlens /score
│   ├── checks.py                        nvidia-smi, port, /health probes
│   └── ui/index.html                    Vanilla HTML/JS, polling progress bar
└── tests/test_pipeline.py               Unit tests (no GPU, mocked subprocesses)
```

---

## Install

```bash
# 1) Install paperprep first (paperlens-reviewing shells out to it):
git clone -b paperprep-only https://github.com/SachinKonan/AutoReviewer.git paperprep
cd paperprep && uv sync --extra gpu && cd ..

# 2) Bring up paperlens serve via paperlens-training-and-inference (separate repo):
#    See https://github.com/SachinKonan/LLaMA-Factory-AutoReviewer/tree/published-branch

# 3) Install this server:
cd paperlens-reviewing
uv venv .venv && source .venv/bin/activate
uv pip install -e .
```

---

## Run

The launcher checks the GPU, brings up `paperlens serve`, and only THEN starts
the reviewing server:

```bash
bash scripts/launch_local.sh
# →  🌐 PaperLens reviewing UI ready at:  http://<host>:8003
```

Open `http://<host>:8003` in a browser, drop a PDF, and watch the stages light up:

```
[ MinerU OCR ] → [ Normalize ] → [ Filter ] → [ Export ShareGPT ] → [ PaperLens score ]
```

Then the verdict card slides in:

```
┌─────────────────────────────────────────────┐
│  Accept                                     │
│  p_accept = 0.8217 · vision / arxiv         │
└─────────────────────────────────────────────┘
```

---

## API surface

| Method | Path                              | Purpose |
|--------|-----------------------------------|---------|
| POST   | `/submit`                         | PDF upload (multipart `file=`, optional form `agent=claude\|codex`); returns `{job_id, stages, agent}` |
| POST   | `/submit_arxiv`                   | JSON `{arxiv_id, main_tex?, agent?}`; live-downloads source from arxiv and routes through the latex path |
| POST   | `/submit_latex`                   | JSON `{path, main_tex?, mode: "latest"\|"history", commits?, n_window?}`; scores a server-side LaTeX dir, optionally across explicit git commits |
| POST   | `/probe_latex`                    | JSON `{path}`; inspect a latex dir (entrypoint guess, tex files, git history with churn) — useful before `mode="history"` |
| GET    | `/status/{job_id}`                | Current `state`, `stage`, fractional progress, `agent_state`, `agent_event_counts`, and (when done) the `result` dict. UI polls this every 1.5 s. |
| GET    | `/jobs/{job_id}/agent_events`     | Query `variant=no_prior\|with_prior&since=N&limit=K`. Returns the next slice of agent transcript events plus terminal predictions. |
| GET    | `/health`                         | Service + upstream health, agent CLI probe (`claude --version`, `codex --version`), stages list |
| GET    | `/examples`                       | Which bundled try-me assets exist on this host |
| GET    | `/jobs`                           | All known job ids (debug) |
| GET    | `/`                               | The HTML UI |

`/status/{job_id}` response shape:

```json
{
  "job_id": "b1c8a7f2…",
  "state": "running",
  "stage": "mineru",
  "stage_index": 0,
  "total_stages": 5,
  "started_at": 1700000000.0,
  "finished_at": null,
  "result": null,
  "stage_log": [{"t": 1700000000.4, "stage": "mineru", "via": "paperprep_state"}]
}
```

When `state="done"`, `result` is:

```json
{
  "decision": "Accept",
  "p_accept": 0.8217,
  "logp_accept": -0.12,
  "logp_reject": -1.84,
  "pred": "Outcome: \\boxed{Accept}",
  "modality": "vision",
  "domain": "arxiv",
  "pdf_name": "yourpaper.pdf",
  "job_dir": ".../paperlensreview_work/<job_id>"
}
```

_Python usage examples for these endpoints live in the umbrella PaperLens
README (`../../README.md#python-usage-examples`) — submitting a PDF / LaTeX
dir / arXiv id, opting into the agentic review, and scoring a custom list
of git commits._

---

## Concurrency

One pipeline at a time per worker. MinerU and `paperlens serve` both want the
GPU; interleaving them produces OOMs. For throughput, run multiple
`paperlensreview` workers behind a queue (e.g. nginx + redis), or run
`paperlens serve` on a dedicated GPU and let the reviewing server fan out.

## Tests

```bash
uv run pytest tests/test_pipeline.py -v
```

All 7 unit tests pass without a GPU (subprocesses + the upstream `/score`
endpoint are mocked).
