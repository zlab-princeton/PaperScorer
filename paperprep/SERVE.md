# paperprep serve — persistent HTTP service

`paperprep serve` runs the same `compile → mineru → normalize → filter → export`
pipeline as `paperprep run`, but behind a small Flask server. Crucially,
MinerU's vLLM engine is loaded **once at startup** as a long-lived child
process (`mineru-vllm-server`) rather than spun up per request — so each
`/prepare` call pays the network round-trip, not the 30–60 s model-load
cost.

This is the prep half of the broader `paperlens` inference stack: one
GPU prepping ShareGPT views, one GPU hosting the acceptance-prediction
model.

---

## Install

```bash
uv sync --extra gpu --extra serve   # gpu for mineru, serve for flask
```

---

## Launch

```bash
srun --gres=gpu:1 -t 24:00:00 --pty bash -lc \
  "uv run paperprep serve \
     --output-dir /tmp/paperprep-serve \
     --port 5000 \
     --mineru-port 30000 \
     --gpu-memory-utilization 0.7"
```

Startup prints `[paperprep serve] MinerU warming up… 35s elapsed` every
10 s and `MinerU ready after 47s (probe=/health)` once vLLM finishes
loading the Qwen2-VL weights. `Ctrl-C` (or `SIGTERM`) tears down the
Flask server and the MinerU subprocess together.

---

## CLI flags

| Flag | Default | What it does |
|---|---|---|
| `--output-dir` (req'd) | — | Per-request artifacts written to `<output-dir>/<request_id>/...` |
| `--host` | `127.0.0.1` | Bind address. Use `0.0.0.0` only behind a trusted network — **there is no auth** |
| `--port` | `5000` | Flask port |
| `--mineru-port` | `30000` | Port for the spawned `mineru-vllm-server` |
| `--mineru-max-pages` | `14` | Cap per-PDF MinerU pages |
| `--min-body-pages` | `6` | Filter threshold (matches `paperprep run`) |
| `--gpu-memory-utilization` | `0.7` | Passed to vLLM. Lower if sharing the GPU |
| `--texlive-bin` | unset | Prepended to `PATH` for the LaTeX compile stage |

---

## Endpoints

### `GET /healthz`

```json
{"status": "ok", "mineru_url": "http://127.0.0.1:30000", "uptime_s": 187}
```

### `POST /prepare` — single paper

```json
{"id": "my_paper_123", "type": "pdf", "path": "/abs/path/foo.pdf",
 "extra": {"title": "...", "abstract": "..."}}
```

### `POST /prepare` — manifest mode (multiple papers in one request)

```json
{"papers": [
  {"id": "a", "type": "pdf",       "path": "/abs/a.pdf"},
  {"id": "b", "type": "latex_dir", "path": "/abs/b_src/", "main_tex": "main.tex"}
]}
```

Both shapes are accepted on the same endpoint. `id` is optional; the
server generates `req_<uuid>` when omitted. `extra` is passed through to
the ShareGPT metadata. Each request blocks until all papers finish —
**the endpoint is synchronous** (no job_id polling).

### `POST /prepare` response (status 200)

```json
{
  "request_id": "20260520-153012-3f4a8c",
  "output_dir": "/tmp/paperprep-serve/20260520-153012-3f4a8c",
  "sharegpt_text_path":   "/tmp/.../sharegpt/text/data.json",
  "sharegpt_vision_path": "/tmp/.../sharegpt/vision/data.json",
  "papers": [
    {"id": "my_paper_123", "status": "ok", "body_pages": 11,
     "normalized_md": ".../normalized/my_paper_123/my_paper_123.md",
     "redacted_pdf":  ".../normalized/my_paper_123/my_paper_123_redacted.pdf"},
    {"id": "bad_paper", "status": "failed", "error": "compile: latexmk timed out"}
  ],
  "elapsed_s": 42.7
}
```

Per-paper failures don't fail the whole request — each paper's status is
reported individually. Whole-request failures (e.g., malformed JSON)
return status 400 / 500 with `{"error": "..."}`.

### `POST /shutdown`

Kills the MinerU subprocess, then exits Flask. Useful for programmatic
teardown (e.g., from a `paperlens` supervisor).

---

## Concurrency

Flask runs with `threaded=True`, so concurrent `/prepare` requests share
the persistent MinerU vLLM server and benefit from vLLM's batching. Each
request gets its own `<request_id>/` subdirectory, so per-paper
artifacts don't race.

---

## Operational notes

- **No `state.jsonl`** — resume / idempotency live in `paperprep run`;
  serve is request-scoped. Outputs persist after the response; cleanup
  is the caller's responsibility (no TTL flag yet).
- **Bind to `127.0.0.1`** unless you've explicitly sandboxed the host.
  `/prepare` reads arbitrary server-local paths and writes anywhere
  under `--output-dir`.
- **Under SLURM**, request enough RAM for vLLM + Qwen2-VL staging.
  Default per-CPU memory allocations are usually too small — the warmup
  OOMs. `--mem=48G` or higher is a safe choice; the integration smoke
  test uses `--mem=80G`.
