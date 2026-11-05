# paperprep

Anonymize, OCR, filter, and ShareGPT-export a batch of academic papers — from
either LaTeX source dirs or PDFs — into the exact same format the PaperScorer
training + inference stack consumes.

The full pipeline in one breath:
**`latex_dir` → strip authors/affiliations/URLs/acks → compile to PDF**
**→ MinerU OCR → strip running headers + refs → redact + rasterize → ShareGPT**.
`pdf` inputs skip the compile step and join at MinerU.

Deeper internals are split out:
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — per-stage mechanics, filter predicates, anonymization mechanics, slurm submitit, output dir layout, ShareGPT shape, edge cases.
- [`SERVE.md`](SERVE.md) — persistent HTTP service mode (one warm MinerU vLLM engine; `/prepare` endpoint).
- [`AGENTS.md`](AGENTS.md) / [`CLAUDE.md`](CLAUDE.md) — pointer doc for LLM/agent navigation.

---

## Install

```bash
git clone https://github.com/zlab-princeton/PaperScorer.git
cd PaperScorer/paperprep
uv sync                  # core (bring your own MinerU outputs)
uv sync --extra gpu      # with MinerU (~17 GB venv: vllm, torch, xformers)
uv sync --extra serve    # adds Flask for `paperprep serve`
```

System deps: **Python ≥ 3.11**, **TeX Live** (for `latex_dir` compile —
point `--texlive-bin` at the dir containing `pdflatex` + `latexmk`),
**CUDA GPU with ≥ 40 GB** for MinerU.

---

## Environment

| Var | Required for | Purpose |
|---|---|---|
| `HF_HOME` | MinerU first run | HF cache root for MinerU model weights (`opendatalab/PDF-Extract-Kit-1.0` + `opendatalab/MinerU2.5-2509-1.2B`). Default `~/.cache/huggingface`; override on slurm to a shared scratch path. |

Everything else (texlive path, slurm partitions, output dir, etc.) is a
CLI flag — see `uv run paperprep run --help`. `TRANSFORMERS_OFFLINE=1`
and `HF_HUB_OFFLINE=1` are exported by paperprep itself before launching
MinerU, so compute nodes don't re-fetch weights mid-job.

One-time warmup so subsequent runs are offline:
```bash
uv run hf download opendatalab/MinerU2.5-2509-1.2B
uv run hf download opendatalab/PDF-Extract-Kit-1.0
```

---

## Quick start

```bash
# Manifest input, single machine
uv run paperprep run \
    --input-manifest examples/paperprep_manifest.jsonl \
    --output-dir ./out \
    --texlive-bin /path/to/texlive/bin/x86_64-linux

# Auto-build manifest from a directory of LaTeX source dirs
uv run paperprep run \
    --input-dir /data/papers/ --input-mode latex_dir \
    --output-dir ./out

# Same but PDFs
uv run paperprep run \
    --input-dir /data/pdfs/ --input-mode pdf \
    --output-dir ./out

# SLURM submission (separate GPU + CPU partitions)
uv run paperprep run \
    --input-manifest manifest.jsonl --output-dir ./out \
    --slurm --gpu-partition gpu --cpu-partition all \
    --chunk-size 50 --max-pages 14 \
    --texlive-bin /path/to/texlive/bin/x86_64-linux
```

---

## Manifest format

JSONL — one paper per line:

```jsonl
{"id": "paper001", "type": "latex_dir", "path": "/data/papers/paper001/"}
{"id": "paper002", "type": "latex_dir", "path": "/data/papers/paper002/", "main_tex": "main.tex"}
{"id": "paper003", "type": "pdf", "path": "/data/papers/paper003.pdf"}
{"id": "paper004", "type": "pdf", "path": "/data/papers/paper004.pdf",
                     "title": "Override Title", "abstract": "Optional abstract override"}
```

| Field | Required? | Notes |
|---|---|---|
| `id` | yes | Unique per manifest. Drives output sub-dir name. |
| `type` | yes | `latex_dir` enters compile; `pdf` skips compile. |
| `path` | yes | Must exist on disk. Validated at manifest load. |
| `main_tex` | opt (`latex_dir` only) | Else auto-detected via `00README.json`'s `toplevelfile`, then first `.tex` containing `\documentclass`. |
| `title` / `abstract` | opt | Used only in the **vision** ShareGPT human turn. Else MinerU first-block heuristic. |
| any other key | opt | Passed through verbatim to the ShareGPT entry's `_metadata`. |

`--input-dir DIR --input-mode {latex_dir|pdf}` auto-builds the manifest by
enumerating immediate sub-dirs (or `*.pdf` files), with `id` = subdir name
or file stem.

---

## Pipeline at a glance

```
manifest.jsonl
     │
     ▼
 ┌──────────────┐                          ┌──────────────┐
 │  pdf input   │                          │ latex_dir    │
 └──────┬───────┘                          │   input      │
        │                                  └──────┬───────┘
        │                                         ▼
        │                            ┌────────────────────────┐
        │                            │ compile                │
        │                            │  • detect .sty / venue │
        │                            │  • L1 anon: strip      │
        │                            │    authors, affil,     │
        │                            │    ack/repro/checklist,│
        │                            │    URLs, leaky thanks  │
        │                            │  • latexmk → PDF       │
        │                            └────────────┬───────────┘
        │                                         │
        ▼                                         ▼
       [ page-count filter: drop PDF with < 4 pages ]
                              │
                              ▼
                  ┌────────────────────────┐
                  │ MinerU OCR             │
                  │  → content_list.json   │
                  │  → middle.json         │
                  │  → images/*.jpg        │
                  └────────────┬───────────┘
                               ▼
              ┌────────────────────────────────┐
              │ normalize                      │
              │  • L2 anon: drop type=header,  │
              │    type=aside_text, refs       │
              │  • PyMuPDF redact authors block│
              │  • rasterize redacted PDF      │
              │    → page_images/page_N.png    │
              └────────────────┬───────────────┘
                               ▼
              ┌────────────────────────────────┐
              │ filter                         │
              │  • body_pages ≥ 6              │
              │  • anonymization marker found  │
              │  • image integrity (non-empty) │
              └────────────────┬───────────────┘
                               ▼
                       [ export ShareGPT ]
                               │
                ┌──────────────┴──────────────┐
                ▼                             ▼
       sharegpt/text/data.json     sharegpt/vision/data.json
                                   (+ page_images/*.png)
```

Compile, mineru, normalize, filter each become a `submitit` array under
`--slurm`; the page-count filter and export run on the driver. Anonymization
happens in **two layers** — once at the LaTeX source level (L1, compile),
once at the OCR/PDF level (L2, normalize) — so `pdf` inputs still get
header/refs scrubbed even though they skipped L1. See
[`ARCHITECTURE.md`](ARCHITECTURE.md#anonymization-details) for the full
mechanics.

---

## Output structure

```
<output-dir>/
  state.jsonl                          # append-only per-(paper, stage) trace; powers --resume
  compiled/
    <id>.pdf                           # only for type=latex_dir (after L1 anon + compile)
  mineru/
    <id>/<pdf_stem>/vlm/
      <pdf_stem>_content_list.json
      <pdf_stem>_middle.json
      <pdf_stem>.md
      images/*.jpg
  normalized/
    <id>/
      <id>.md                          # post-L2 normalized markdown (refs + headers stripped)
      <id>_content_list.json           # filtered content_list
      <id>_redacted.pdf                # PyMuPDF whiteout PDF; falls back to original on failure
      page_images/page_N.png           # 150 DPI raster, 1-indexed
  sharegpt/
    text/data.json                     # all surviving papers' text entries
    vision/data.json                   # same papers, vision entries (with images[])
  logs/<run-id>/
    {compile,mineru,normalize,filter}/<job_id>_0.{out,err,result.pkl}
```

Both ShareGPT files are arrays of `[{conversations: [system, human], _metadata, images?: [...]}]` —
inference-only entries (no `gpt` turn).

---

## Common knobs

| Flag | Default | What it does |
|---|---|---|
| `--max-pages N` | `14` | MinerU per-PDF page cap. Matches the production arxiv pipeline. Bump for long-paper batches (≈ 2× MinerU wall). |
| `--min-body-pages N` | `6` | Drop papers with detected body length below this. Matches the production `hf_filter` gate (cuts ACL Short / AAAI Student Abstracts). |
| `--dpi N` | `150` | Page-image rasterization DPI. |
| `--compile-timeout SECONDS` | `600` | Per-paper latexmk wall-time cap. |
| `--task-prompt STRING` | the prod multi-venue generic prompt | Override the user prompt embedded in the ShareGPT human turn. |
| `--chunk-size N` | `50` | Papers per submitit array task. MinerU is internally clamped to `min(N, 20)`. |
| `--stage-overrides 'mineru.timeout_min=480,compile.mem_gb=16'` | `""` | Per-stage submitit overrides — any field on `StageConfig` (partition, timeout_min, cpus_per_task, mem_gb, gpus_per_node, array_parallelism, extra_slurm). |
| `--no-resume` | resume ON | Force re-run of every paper through every stage. By default, papers with `status=ok` rows in `state.jsonl` are skipped. |

Full flag list: `uv run paperprep run --help`. Per-stage submitit defaults
(partitions, timeouts, mem, gpus, array parallelism) are in
[`ARCHITECTURE.md`](ARCHITECTURE.md#slurm-submitit).

---

## Serving as a persistent HTTP service

For latency-sensitive single-paper requests (e.g. behind the
[`paperscorer-reviewing`](../tools/paperscorer-reviewing) UI), `paperprep serve`
holds MinerU's vLLM engine warm and exposes a synchronous `/prepare`
endpoint. See [`SERVE.md`](SERVE.md).

---

## Library usage

Every stage is a free function importable from Python — useful for
embedding paperprep into other pipelines or running a single stage
without the full driver. See
[`ARCHITECTURE.md`](ARCHITECTURE.md#library-usage) for the full
import surface; quick taste:

```python
from paper_anonymizer.paperprep import manifest
from paper_anonymizer.paperprep.stages import compile as stage_compile

entries = manifest.load_manifest(Path("manifest.jsonl"))
results = stage_compile.run_chunk(entries, Path("./out"),
                                  texlive_bin="/path/to/texlive/bin/...")
```

---

## Testing

Unit tests run on every push via GitHub Actions
(`.github/workflows/tests.yml`); the GPU-required integration suites
(`tests/integration/test_golden_e2e_{pdf,latex}.py`, ~20–40 min on an
L40) are opt-in for maintainers via `pytest -m gpu` / `-m slow`. See
[`ARCHITECTURE.md`](ARCHITECTURE.md#testing) for what the golden suites
assert and the fixture layout.

---

## Using paperprep in another project

paperprep ships as the `paperprep/` directory of
[zlab-princeton/PaperScorer](https://github.com/zlab-princeton/PaperScorer).
To install it into another environment without cloning the whole repo:

```bash
uv pip install "paperprep @ git+https://github.com/zlab-princeton/PaperScorer.git#subdirectory=paperprep"
```
