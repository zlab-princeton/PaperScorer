# paperprep — architecture

Deep-dive companion to [`README.md`](README.md). Quickstart, install, env
vars, manifest format, and the high-level pipeline diagram live in the
README; this document is for someone modifying paperprep or debugging
why a paper got dropped.

---

## Table of contents

1. [Pipeline stages](#pipeline-stages)
2. [Filter predicates](#filter-predicates)
3. [Anonymization details](#anonymization-details)
4. [SLURM submitit](#slurm-submitit)
5. [Heterogeneous-cluster handling](#heterogeneous-cluster-handling)
6. [Resume + idempotency](#resume--idempotency)
7. [Monitoring + tqdm](#monitoring--tqdm)
8. [ShareGPT output shape](#sharegpt-output-shape)
9. [Library usage](#library-usage)
10. [Testing](#testing)
11. [Edge cases + known limitations](#edge-cases--known-limitations)
12. [Project structure](#project-structure)

---

## Pipeline stages

### Stage 1 — compile

`src/paper_anonymizer/paperprep/stages/compile.py`

For each paper with `type=latex_dir`:

1. Copy the source dir into a `tempfile.TemporaryDirectory` (we never
   mutate the user's input).
2. For every `*.tex` in the staging dir: apply `apply_venue_toggle(text)`
   from `core/venue_toggle.py`. Universal-anonymization pass:
   - Replaces the body of `\author{}`, `\institute{}`, `\affiliations{}`,
     `\thanks{}` with `Anonymous Submission` / `Anonymous Institute`.
   - Comments out `\*finalcopy` macros (ICLR).
   - Strips `[final]` / `[accepted]` options from NeurIPS / ICML /
     AISTATS package imports.
   - Adds `[review]` to CVPR / ICCV / ECCV / ACL-family `\usepackage`.
   - Adds `[submission]` to AAAI 2023+ / COLM.
3. Call `scrub_tex_dir(staging)` from `core/tex.py`. Recursively visits
   every `*.tex` and:
   - Removes `\section{Acknowledg...}`, `\section{Reproducibility}`,
     `\section{Checklist}` and their contents until the next section.
   - Deletes `\footnote{...}` whose body matches a URL / email /
     "equal contribution" pattern (skips footnotes in protected
     author/institute ranges).
   - Removes `\url{...}` commands; unwraps `\href{url}{text}` → `text`;
     strips bare `https://...` URLs.
4. Detect the LaTeX template venue against `configs/templates.yaml`. If
   no pattern matches, **emit a warning and continue** — papers with
   non-standard classes (Meta's `fairmeta.cls`, custom lab templates)
   still get processed; they just lose venue-specific toggles on
   un-toggleable packages.
5. Resolve the main tex: manifest's `main_tex` field if given, else
   `00README.json`'s `toplevelfile`, else the first `*.tex` whose body
   contains `\documentclass` (shallowest path first).
6. `run_latexmk(main_tex, timeout=--compile-timeout, texlive_bin=...)`:
   `latexmk -pdf -interaction=nonstopmode -file-line-error -f`.
   Force-mode means the compile keeps going on recoverable errors; the
   resulting PDF is accepted if it exists, regardless of exit code.
7. Copy the resulting PDF to `<output_dir>/compiled/<id>.pdf`. Capture
   the page count via `pdfinfo`.
8. Record `StageResult(stage="compile", status="ok",
   output_path=..., extra={page_count, toggle_changes, scrub_stats,
   template_venue, template_year, warning})`.

Failure modes (`status="failed"`):
- `copytree failed: <err>`
- `scrub_tex_dir failed: <err>`
- `no main .tex found (00README.json missing or no \documentclass)`
- `compile produced no pdf`
- `pdf copy failed: <err>`

`type=pdf` entries return immediately with `status="skipped"` (the PDF is
forwarded as-is to the next stage).

### Stage 2 — page-count filter (inline)

`src/paper_anonymizer/paperprep/stages/detect.py`

Inline driver-side step. For each `(id, pdf_path)`:
- Read page count via `pdfinfo`.
- If `page_count < 4` → drop (`status="failed"`, `error="page_count=N < 4"`).
- If `pdfinfo` couldn't read the file → `status="failed"`,
  `error="pdfinfo could not read page count"`.

### Stage 3 — mineru

`src/paper_anonymizer/paperprep/stages/mineru.py`

For each surviving PDF, calls `run_mineru_single(pdf_path, out_dir,
max_pages=--max-pages)` from `core/ocr.py`. Under the hood:

```
mineru -p <pdf> -o <out_dir> -b vlm-vllm-engine --end <max_pages>
```

with `TRANSFORMERS_OFFLINE=1` and `HF_HUB_OFFLINE=1` (the compute nodes
must not re-download models — they're pre-staged at the HF cache).

Each chunk = one submitit task. Default `array_parallelism=20` for the
MinerU stage (vs 80 for CPU stages) — limits concurrent GPU jobs to
avoid hammering shared GPUs and to give the MinerU model time to load
before its 45-min idle kill triggers.

Output per paper:
`<output_dir>/mineru/<id>/<inner_dir>/vlm/<inner_dir>_content_list.json`,
`_middle.json`, `.md`, `images/*.jpg`. `inner_dir` is the PDF stem (a
MinerU artifact — paperprep handles it via the `find_content_list` and
`find_middle_json` glob helpers).

Failure modes:
- `run_mineru_single raised: <err>`
- `result.success is False, error: <err>`
- `content_list.json not found in mineru output`

### Stage 4 — normalize

`src/paper_anonymizer/paperprep/stages/normalize.py`

For each paper that succeeded at mineru:

1. Load `content_list.json`.
2. `filter_content_list(content_list, FilterMode.ANONYMOUS)` — strips
   everything after the References heading, removes `type="header"` and
   `type="aside_text"` blocks. Preserves title + abstract + body.
3. Render markdown via `render_md_from_content_list(filtered)` from
   `dataset/canonical_format.py`:
   - Iterates blocks in order.
   - `type="text"` → `#`-headers based on `text_level`.
   - `type="equation"` → `$$...$$`.
   - `type="table"` → raw HTML.
   - `type="image"` → `![](<path>)`.
   - `type="list"` → bullet items.
4. Write filtered MD → `<output_dir>/normalized/<id>/<id>.md`.
5. Write filtered content list →
   `<output_dir>/normalized/<id>/<id>_content_list.json`.
6. PDF redaction (best-effort):
   - If `middle.json` is present: compute `(items_to_whiteout, page_end)`
     via `compute_anonymous_whiteout(content_list)`. Collects every
     `type="header"`, `type="aside_text"`, plus mid-page references
     content if refs and body share a page.
   - Convert normalized bboxes (0–1000 range) → PDF points using
     `load_page_sizes(middle_json)`.
   - `redact_pdf(...)` — PyMuPDF `add_redact_annot` white fill,
     `apply_redactions`, then `delete_pages` beyond `page_end`.
   - On any throw: log a warning and **fall back to the unredacted PDF**.
     The paper still goes through; pages may include identity markers.
7. `pdf_to_images(redacted_pdf, page_images_dir, dpi=--dpi)` →
   1-indexed `page_N.png` at the configured DPI.

Failure modes:
- `failed to load content_list: <err>`
- `filter_content_list failed: <err>`
- `render_md_from_content_list failed: <err>`
- `pdf_to_images failed: <err>`

Redaction-only failures are logged but **do not fail the stage** — they
downgrade to "rendered from unredacted PDF".

### Stage 5 — filter

`src/paper_anonymizer/paperprep/stages/filter.py`

For each normalized paper:

1. **Body-pages cascade** (mirrors the production arxiv
   `scripts/analyze_body_pages.py:_analyze`):
   1. First heading matching `REFERENCES_PATTERN`
      (`^\s*(references?|bibliography)\b`, case-insensitive): `body_pages
      = page_idx`, method `references_heading`.
   2. Else first block with `sub_type == "ref_text"`: `body_pages =
      page_idx`, method `ref_text_walkback`.
   3. Else first heading containing `conclusion` (case-insensitive):
      `body_pages = page_idx + 1`, method `conclusion_buffer`.
   4. Else `body_pages = max(page_idx) + 1`, method `total_pages_fallback`.
2. **Anonymization check** — regex
   `\banonymous\b|confidential\s+review|under\s+review|submission\s*#|blind\s+submission`
   (case-insensitive). Scans:
   - All `type="header"` blocks (running headers).
   - First 10 text blocks on `page_idx=0` (catches "Under review at
     ICLR" if it's the first text on page 1).
3. **Image integrity** — verify `<output_dir>/normalized/<id>/page_images/`
   exists, contains at least one `page_*.png`, and the first
   `body_pages` of them all have non-zero size.

### Stage 6 — export

`src/paper_anonymizer/paperprep/stages/export.py`

For each surviving paper, build BOTH a text and a vision ShareGPT entry:

**Text entry** —
`paper_text = build_text_paper_text(content_list_path, body_pages=N)`.
Renders MD via `render_md_from_content_list(filtered, body_pages=N)`,
truncated to `page_idx < body_pages`, then strips literal `<image>`
tokens (e.g. VLM example papers that contain the literal text).
`create_sharegpt_conversation(paper_text, label=None, task_prompt=...)`
→ entry with `[system, human]` turns only.

**Vision entry** —
`title, abstract = _extract_title_abstract(content_list, extra)` —
prefers manifest extras, else first `text_level=1` block (title) and
first text starting with "abstract".
`paper_text = build_vision_paper_text(title, abstract, n_images)` →

```
# {title}

## Abstract
{abstract}

<image> <image> ... <image>      ← n_images of them, space-separated
```

Page images = `numerical_page_sort(glob page_*.png)[:body_pages]`
(numerical sort handles `page_2.png < page_10.png`). Entry includes
`images=[...]` with absolute on-disk paths.

Both entries get `_metadata = {id, body_pages, n_images?,
**manifest_extras}`.

Output:
- `<output_dir>/sharegpt/text/data.json` — all surviving papers' text entries
- `<output_dir>/sharegpt/vision/data.json` — same papers, vision entries

Export is a driver-side pass; not a submitit job.

---

## Filter predicates

Every drop gate with its threshold and implementation site:

| Where | Predicate | Threshold | Implementation |
|---|---|---|---|
| compile | latexmk produced no PDF | n/a | `core/latex_compile.py:run_latexmk` |
| compile | no main `.tex` found | n/a | `core/latex_compile.py:find_main_tex` |
| page_count_filter | PDF page count too low | `< 4` | `paperprep/stages/detect.py:filter_short_pdfs` |
| page_count_filter | `pdfinfo` couldn't read | n/a | same |
| mineru | mineru CLI exited non-zero | n/a | `core/ocr.py:run_mineru_single` |
| mineru | content_list.json absent | n/a | same |
| normalize | content_list load error | n/a | `paperprep/stages/normalize.py:_normalize_one` |
| normalize | filter/render/rasterize error | n/a | same |
| filter | body_pages below threshold | `< 6` (default) | `paperprep/stages/filter.py:_filter_one` |
| filter | no anonymization marker | n/a | same |
| filter | no `page_*.png` files | n/a | same |
| filter | any of first `body_pages` images is 0 bytes / missing | n/a | same |

### Difference from production gates

paperprep's filter is strictly equivalent to the production arxiv
pipeline's `stage_hf_filter` **except** that paperprep does NOT apply
the **per-venue p95 reject body_pages cap**. That cap is a balancing
decision for the production training set; for paperprep's use case
(arbitrary user batches) it doesn't apply.

---

## Anonymization details

Two layers stack — `pdf` inputs only get Layer 2; `latex_dir` inputs get
both.

### Layer 1 — LaTeX source manipulation (compile stage)

For `type=latex_dir` only. `apply_venue_toggle` + `scrub_tex_dir` run
against every `*.tex` in the staged temp dir BEFORE compilation, so the
compiled PDF itself has no author identity, no acknowledgements, no
reproducibility/checklist sections, no leaky footnotes, no URLs.

| Source | Transformed to |
|---|---|
| `\author{Jane Doe and John Smith}` | `\author{Anonymous Submission and Anonymous Submission}` |
| `\affiliation[1]{University X}` | `\affiliation[1]{Anonymous Institute}` |
| `\thanks{Equal contribution}` | `\thanks{Anonymous Submission}` (only if body matches a leak pattern) |
| `\iclrfinalcopy` | `% \iclrfinalcopy` |
| `\usepackage[final]{neurips_2024}` | `\usepackage{neurips_2024}` |
| `\usepackage{cvpr}` | `\usepackage[review]{cvpr}` |
| `\section{Acknowledgments}\n<body>\n\section{...}` | next-section header preserved, ack content removed |
| `\section{Reproducibility Statement}\n<body>` | removed |
| `\section{Checklist}\n<body>` | removed |
| `\url{https://github.com/foo}` | removed (URL stripped, surrounding sentence preserved) |
| `\href{https://github.com/foo}{our codebase}` | `our codebase` |
| `\footnote{Code: https://github.com/foo}` | removed entirely (leaky footnote) |
| `\footnote{See Section 3}` | **kept** (no leak pattern) |

### Layer 2 — content_list / PDF redaction (normalize stage)

After MinerU OCR, the structured content_list drives:
- **MD render**: drop `type=header` (running headers) + `type=aside_text`
  (review-marker ribbons) before rendering.
- **PDF redaction**: PyMuPDF white-out of the same blocks' bbox regions,
  then `delete_pages` beyond `body_pages + 1`.

For `type=pdf` inputs, only Layer 2 runs. Callers are responsible for
supplying anonymized PDFs — or accepting that author identity leaks if
the PDF isn't already anonymized.

The anonymization regex check in the filter stage acts as a final
safety net: papers without at least one anonymization marker
("Anonymous", "Under review", "Submission #", "Confidential review",
"Blind submission") on page 0 or in any header are dropped.

---

## SLURM submitit

When `--slurm` is set, each stage is submitted as a
`submitit.AutoExecutor` array of chunked tasks. Per-stage defaults live
in `src/paper_anonymizer/paperprep/submit.py:DEFAULT_STAGE_CONFIGS`:

| Stage | partition routing | timeout | cpus | mem | gpus | array parallelism |
|---|---|---|---|---|---|---|
| `compile`   | `--cpu-partition` | 120 min |  4 |  8 GB | 0 | 80 |
| `mineru`    | `--gpu-partition` | 240 min |  4 | 64 GB | 1 | 20 |
| `normalize` | `--cpu-partition` | 120 min |  8 | 16 GB | 0 | 80 |
| `filter`    | `--cpu-partition` |  30 min |  4 |  8 GB | 0 | 80 |
| `export`    | `--cpu-partition` |  30 min |  4 |  8 GB | 0 | (driver-local) |

The driver waits on each stage's array to complete before submitting
the next. `state.jsonl` records per-(paper, stage) status incrementally
so `--resume` works across cross-stage restarts.

### Per-stage overrides

```
--stage-overrides 'mineru.timeout_min=480,compile.mem_gb=16,mineru.array_parallelism=10'
```

Comma-separated `stage.key=value`. Values coerce to `int` if possible,
else `float`, else string. Any `StageConfig` field is overridable:
`partition`, `timeout_min`, `cpus_per_task`, `mem_gb`, `gpus_per_node`,
`array_parallelism`, `extra_slurm`.

### Chunking

`--chunk-size N` (default 50). Each chunk = one submitit array task. The
MinerU stage is internally clamped to `min(--chunk-size, 20)` because
GPU tasks scale poorly past ~20 concurrent jobs on a shared cluster
(model loading stampedes NFS).

---

## Heterogeneous-cluster handling

The design assumes a single SLURM cluster where the GPU partition
(`--gpu-partition`) and the CPU partition (`--cpu-partition`) are
reachable from the same login node. Both flags accept the same partition
name on homogeneous setups:

```bash
# heterogeneous (separate GPU + CPU pools)
--gpu-partition gpu --cpu-partition cpu

# single-partition cluster
--gpu-partition all --cpu-partition all
```

Cluster-specific tuning via `--stage-overrides`:

```bash
--gpu-partition gpu --cpu-partition cpu \
--stage-overrides 'mineru.array_parallelism=8,mineru.mem_gb=80,compile.timeout_min=180'
```

There is **no multi-cluster fallback** (no `--partition pa,pb,pc` retry
list). Wrap paperprep in a shell that catches failed runs and retries
with a different partition if you need that.

---

## Resume + idempotency

Every per-(paper, stage) outcome is appended to
`<output_dir>/state.jsonl`:

```jsonl
{"id": "paper001", "stage": "compile", "status": "ok",
 "output_path": "/.../compiled/paper001.pdf",
 "started_at": 1778601628.027, "ended_at": 1778601628.027,
 "extra": {"page_count": 19, "toggle_changes": 9, "scrub_stats": {...},
           "template_venue": null, "template_year": null,
           "warning": "no LaTeX template venue detected ..."}}
```

`--resume` is on by default (`--no-resume` disables). It skips any paper
that already has a `status=ok` row for the upcoming stage. Rerunning
paperprep with the same `--output-dir` is safe and incremental — if
MinerU OOM'd on 3 papers in a 100-paper batch, just rerun and the 97
successes are skipped.

---

## Monitoring + tqdm

**Driver-side bars** (when `--slurm`): one tqdm bar per stage, polling
each submitit Job state every 2 s. Terminal states (`COMPLETED`,
`FAILED`, `CANCELLED`, `TIMEOUT`, `OUT_OF_MEMORY`) advance the bar.

**Worker-side bars**: every per-paper loop inside a chunk uses
`paper_iter()` (tqdm wrapper). Writes go to the submitit log file at
`<logs_dir>/<stage>/<job_id>_0.log`; tail it from the login node.

**End-of-run summary**: paperprep prints a stage-by-(ok/skipped/failed)
tally to stdout from `state.jsonl`:

```
[paperprep] per-stage summary:
  compile              ok=3 skipped=3
  mineru               ok=6
  normalize            ok=6
  filter               ok=6
  export               ok=6
```

---

## ShareGPT output shape

Entries are **inference-only**: `[system, human]` conversation roles
only, no `gpt` turn (user-supplied papers don't have known labels).
Downstream consumers must tolerate `len(conversations) == 2`.

### Text entry

```json
{
  "conversations": [
    {"from": "system", "value": "<SHAREGPT_SYSTEM_PROMPT>"},
    {"from": "human",  "value": "<task_prompt>\n\n<paper_text>"}
  ],
  "_metadata": {"id": "paper001", "body_pages": 12,
                "...": "manifest extras (except title/abstract)"}
}
```

### Vision entry

```json
{
  "conversations": [
    {"from": "system", "value": "<SHAREGPT_SYSTEM_PROMPT>"},
    {"from": "human",  "value": "<task_prompt>\n\n# {title}\n\n## Abstract\n{abstract}\n\n<image> <image> ... <image>"}
  ],
  "_metadata": {"id": "paper001", "body_pages": 12, "n_images": 12, "...": "manifest extras"},
  "images": [
    "<absolute_path>/normalized/paper001/page_images/page_1.png",
    "<absolute_path>/normalized/paper001/page_images/page_2.png"
  ]
}
```

### Prompts (in `core/constants.py`)

`SHAREGPT_SYSTEM_PROMPT`:
```
You are an expert academic reviewer tasked with evaluating research papers.
```

`SHAREGPT_USER_PROMPT_GENERIC` (default; override via `--task-prompt`):
```
I am giving you a paper submitted to a top machine-learning venue. Predict its acceptance outcome.
 - Your answer will either be: \boxed{Accept} or \boxed{Reject}
 - Note: typical top-tier ML venues have ~25-30% acceptance rates
```

### Format equivalence with production

For arxiv-source papers, paperprep's ShareGPT entries are functionally
equivalent to the production `balanced_per_venue_{text,vision}_*` JSONs:
- Identical system + task prompt.
- Identical paper_text body content (modulo MinerU non-determinism).
- Identical `<image>`-token count alignment with `images[]`.
- Only structural difference: paperprep omits the `gpt` turn
  (`Outcome: \\boxed{Accept|Reject}`).

For ICLR-source papers, paperprep's body matches the arxiv-style
filtered output (refs + everything after refs dropped), NOT the ICLR
Level-2 normalization (which keeps the appendix). By design — the spec
called for paperprep to mirror the arxiv filters.

---

## Library usage

Every stage is a free function importable from Python:

```python
from pathlib import Path

from paper_anonymizer.paperprep import manifest, state, submit
from paper_anonymizer.paperprep.stages import (
    compile as stage_compile,
    detect as stage_detect,
    mineru as stage_mineru,
    normalize as stage_normalize,
    filter as stage_filter,
    export as stage_export,
)
from paper_anonymizer.dataset.sharegpt import create_sharegpt_conversation

# Build entries
entries = manifest.load_manifest(Path("manifest.jsonl"))
# or
entries = manifest.build_manifest_from_dir(Path("/data/papers"), "latex_dir")

# Run a single stage locally
compile_results = stage_compile.run_chunk(
    entries, Path("./out"), timeout=600,
    texlive_bin="/path/to/texlive/bin/x86_64-linux",
)

# State log
log = state.StateLog(Path("./out/state.jsonl"))
log.append_many(compile_results)
print(log.summarize())

# Submit a stage to slurm
jobs = submit.submit_stage(
    "mineru",
    items=[("p1", Path("p1.pdf"))],
    fn=lambda chunk: stage_mineru.run_chunk(chunk, Path("./out"), max_pages=14),
    cpu_partition="cpu", gpu_partition="gpu",
    chunk_size=20, logs_dir=Path("./logs"),
)
# jobs is list[submitit.Job]; call j.result() to fetch StageResults

# Build a single ShareGPT entry directly (inference-only)
entry = create_sharegpt_conversation(
    paper_text="My paper content...",
    label=None,                           # None → no gpt turn
    metadata={"id": "x"},
    images=["page_1.png", "page_2.png"],
)
```

Core utilities (LaTeX compile, MinerU runner, content filtering, PDF
redaction, ShareGPT builders) are importable from
`paper_anonymizer.core.*` and `paper_anonymizer.dataset.*`. Submitit can
wrap any `run_chunk` for cluster execution via `submit.submit_stage(...)`.

---

## Testing

### Unit tests (fast, no GPU)

```bash
uv run pytest tests/unit/ --no-header -q
# expect: 73 passed, 26 skipped
```

The 26 skipped are in `test_venue_toggle.py` — those fixtures predate a
refactor that made the `venue` argument a no-op (universal heuristics
apply now). The implementation is exercised end-to-end via the compile
stage and through `test_tex_scrub.py`.

CI: `.github/workflows/tests.yml` runs `pytest tests/unit/` on every
push.

### Integration tests (golden-output diff vs production)

Two opt-in suites in `tests/integration/` certify paperprep's output
matches the production arxiv + ICLR pipelines for the same input PDFs.
Inputs + production ground-truth ship in the repo
(`tests/fixtures/golden/`, ~44 MB) so the test is cluster-portable.

```bash
# PDF-only golden test (~20 min on L40, requires --extra gpu)
uv run pytest tests/integration/test_golden_e2e_pdf.py -m gpu --no-header -s

# Full LaTeX + PDF golden test (~40 min, includes everything from -m gpu)
uv run pytest tests/integration/test_golden_e2e_latex.py -m slow --no-header -s
```

What the integration tests assert:

| Concern | Threshold |
|---|---|
| Every paper through every stage with `status=ok` | hard |
| Redacted PDF produced | exists + > 0 bytes |
| arxiv MinerU block-count vs production | ±5 blocks |
| arxiv MinerU type-set vs production | identical (modulo content-dependent `code`/`page_footnote`/`header`) |
| arxiv MinerU text-block overlap vs production | ≥ 85% |
| arxiv normalized MD similarity vs production | ≥ 0.90 |
| ICLR normalized MD similarity vs production L2 | ≥ 0.60 (design-difference appendix) |
| arxiv ShareGPT text-body similarity vs production | ≥ 0.90 |
| ShareGPT roles `[system, human]` (no gpt turn) | exact |
| arxiv vision image count vs production | exact |
| **(LaTeX-only)** Compile produces PDF, page count within ±1 of production | hard |
| **(LaTeX-only)** `apply_venue_toggle` rewrote ≥ 1 LaTeX command | `toggle_changes > 0` |
| **(LaTeX-only)** `scrub_tex_dir` ran (7-key stats dict) | schema check; all-zero stats valid for clean papers |

Fixture layout:

```
tests/fixtures/golden/
  inputs/
    iclr_<sid>.pdf  (×3)
    arxiv_<aid>.pdf  (×3)
    arxiv_<aid>.gz   (×3)                # extracted at test time into tmp_path
  ground_truth/
    arxiv_<aid>/
      mineru_content_list.json
      mineru_middle.json
      normalized.md
      production_pdf_page_count.txt
      sharegpt_text_entry.json
      sharegpt_vision_entry.json
    iclr_<sid>/
      normalized_l2.md
      sharegpt_text_entry.json
```

Set `PAPERPREP_TEXLIVE_BIN` to override the TeX Live path used by the
LaTeX integration test.

---

## Edge cases + known limitations

### MinerU `max_pages` cap

Default `--max-pages 14`. Papers longer than 14 pages have their tail
truncated. For ICLR/NeurIPS-style papers (8 body + 4 refs + appendix),
14 covers body + refs but loses most of the appendix. **Intentional** —
production uses the same cap. Bump via `--max-pages 30` for long-paper
batches (~2× MinerU wall per paper).

### Title casing for arxiv papers

Production sources the title from arxiv metadata
("Transformers without Normalization"). paperprep takes it from
MinerU's first `text_level=1` block, which may be the PDF's running
header in caps ("TRANSFORMERS WITHOUT NORMALIZATION"). The vision human
turn reflects this.

**Workaround**: supply `"title": "..."` in the manifest extras.

### Non-standard abstract macros

Papers using non-`\section{Abstract}` macros (e.g. `\abstract{...}`)
may have MinerU emit the abstract content without a clear header.
`_extract_title_abstract` falls back to empty abstract. The **text-mode**
entry is unaffected (full body renders); only the vision entry's
`## Abstract\n` block becomes empty.

**Workaround**: supply `"abstract": "..."` in the manifest extras.

### PyMuPDF / MinerU version drift

PDF rasterization is deterministic for a given PyMuPDF version. MinerU
content extraction is deterministic for a given mineru + vLLM version,
but vLLM's batched inference can show small non-determinism (1–5% of
text blocks may re-chunk between runs). For exact byte-equivalence with
production, pin `mineru[core,vllm]==2.6.2` (what the `[gpu]` extra
pins).

### Mid-sentence cropping at body→refs boundary

The body-pages cascade clips at the page_idx of the References heading.
If the last body paragraph spills onto the same page as the refs
heading, it's dropped at its truncation point — sometimes mid-sentence.
Same behavior as production. Affects ~30–50% of papers, ~one lost
sentence per affected paper.

### Compile-stage `template_venue` warning

Papers using custom LaTeX class files (Meta's `fairmeta.cls`, custom
lab templates) won't match any pattern in `configs/templates.yaml`.
paperprep logs `"no LaTeX template venue detected"` and proceeds.
Venue-specific toggles like adding `[review]` to `\usepackage{cvpr}`
won't fire — that's fine, those papers' classes don't use those
packages.

### No GPU fallback for MinerU

If the GPU partition is full or the job is preempted, MinerU jobs fail.
`state.jsonl` records the failure; `--resume` retries. No automatic CPU
fallback (MinerU's vLLM backend is GPU-only).

### Single SLURM cluster only

`--gpu-partition` and `--cpu-partition` must refer to partitions on the
same SLURM cluster.

### `mineru --version` cold-start

`mineru --version` initializes ONNX under the hood and takes ~6 s cold.
The `have_mineru()` check in the integration-test conftest uses a 60 s
timeout to be safe on slow / cold cluster nodes — if you see all tests
silently `SKIPPED`, that's the canary.

---

## Project structure

```
paperprep/
├── pyproject.toml                          # 4 deps + [gpu] + [serve] extras; paperprep = entry point
├── uv.lock
├── README.md                               # quickstart
├── ARCHITECTURE.md                         # this file
├── SERVE.md                                # `paperprep serve` HTTP service docs
├── AGENTS.md / CLAUDE.md                   # LLM/agent pointer
├── configs/templates.yaml                  # LaTeX venue template detection regexes
├── examples/paperprep_manifest.jsonl       # sample input manifest
├── src/paper_anonymizer/
│   ├── __init__.py                         # re-exports FilterMode, filter_content_list
│   ├── core/
│   │   ├── constants.py                    # SHAREGPT_*_PROMPT, REFERENCES_PATTERN
│   │   ├── filters.py                      # FilterMode + filter_content_list
│   │   ├── latex_compile.py                # extract_source, find_main_tex, run_latexmk
│   │   ├── ocr.py                          # run_mineru, run_mineru_single
│   │   ├── pdf.py                          # redact_pdf, pdf_to_images
│   │   ├── pdf_filter.py                   # compute_anonymous_whiteout
│   │   ├── tex.py                          # scrub_tex, scrub_tex_dir
│   │   └── venue_toggle.py                 # apply_venue_toggle
│   ├── dataset/
│   │   ├── canonical_format.py             # build_{text,vision}_paper_text, numerical_page_sort
│   │   └── sharegpt.py                     # create_sharegpt_conversation
│   └── paperprep/
│       ├── cli.py                          # argparse: paperprep run
│       ├── serve.py                        # paperprep serve (HTTP)
│       ├── manifest.py                     # JSONL parser + validator
│       ├── progress.py                     # tqdm helpers
│       ├── state.py                        # StateLog (append-only JSONL)
│       ├── submit.py                       # submitit AutoExecutor wrapper + StageConfig
│       └── stages/
│           ├── compile.py                  # LaTeX → anonymized PDF
│           ├── detect.py                   # page-count filter
│           ├── mineru.py                   # PDF → content_list + middle.json + page imgs
│           ├── normalize.py                # MinerU → md + redacted PDF + page_N.png
│           ├── filter.py                   # body_pages + anon + image integrity gates
│           └── export.py                   # text + vision ShareGPT JSONs
└── tests/
    ├── unit/                               # CI; 73 pass, 26 skipped (venue_toggle stale fixtures)
    └── integration/                        # maintainer-only; -m gpu / -m slow
        └── ../fixtures/golden/             # checked-in inputs + production ground-truth (~44 MB)
```
