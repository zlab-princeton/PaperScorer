# Agent / LLM Guide — paperprep

Entry point for any agent (Claude Code, Codex, Cursor, etc.) landing in
the repo. `README.md` is for humans installing + running paperprep; this
file is for agents navigating the code. `CLAUDE.md` is a symlink to this
file — both CLIs pick up the same content from the cwd.

---

## What this repo is

paperprep takes a batch of papers (LaTeX source dirs or PDFs) and emits
the exact ShareGPT format the PaperLens training + inference stack
consumes. End-to-end: anonymize → compile → MinerU OCR → redact +
rasterize → filter → export.

Three docs:
- [`README.md`](README.md) — install, env vars, quickstart, manifest format, pipeline diagram, output layout.
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — per-stage internals, filter predicates, anonymization mechanics, slurm submitit, ShareGPT shape, edge cases.
- [`SERVE.md`](SERVE.md) — persistent HTTP service mode (`paperprep serve`).

---

## Where to start a task

| Task | Start here |
|---|---|
| One-shot run on a batch | `README.md` → Quick start. `paperprep run --input-manifest ... --output-dir ...`. |
| Embed paperprep behind an HTTP endpoint | [`SERVE.md`](SERVE.md). |
| Debug "why did paper X get dropped" | `<output_dir>/state.jsonl` — find the row with `status=failed`. Then [`ARCHITECTURE.md` § Filter predicates](ARCHITECTURE.md#filter-predicates) for which gate matched. |
| Add a new anonymization rule | `core/venue_toggle.py` (LaTeX source level) or `core/pdf_filter.py` (post-OCR redaction). See [`ARCHITECTURE.md` § Anonymization details](ARCHITECTURE.md#anonymization-details). |
| Add a new stage to the pipeline | New module under `src/paper_anonymizer/paperprep/stages/`. Wire it in `cli.py`'s stage list + `submit.py:DEFAULT_STAGE_CONFIGS`. |
| Embed paperprep in another tool | `paper_anonymizer.paperprep.stages` exports every stage as a free function. See [`ARCHITECTURE.md` § Library usage](ARCHITECTURE.md#library-usage). |
| Change the ShareGPT prompts or schema | `core/constants.py` (prompts), `dataset/sharegpt.py` (entry shape), `dataset/canonical_format.py` (body rendering). **This affects PaperScorer training data downstream** — grep `tools/paperscorer-reviewing` + `paperscorer-training-and-inference` before merging. |

---

## Hard rules

1. **Two anonymization layers, two input types.** `latex_dir` inputs get **both** layers (L1 LaTeX scrub + L2 post-OCR redaction). `pdf` inputs only get L2. Anything that breaks this assumption (e.g. running L1 on a pre-compiled PDF) silently no-ops, doesn't error — be careful.
2. **MinerU stage is GPU-bound and array-parallelism-clamped to 20.** Don't raise the clamp unless you've verified the cluster's NFS won't stampede on model load. See [`ARCHITECTURE.md` § Chunking](ARCHITECTURE.md#chunking).
3. **`state.jsonl` is append-only.** Never edit it by hand to "fix" a failed paper — rerun with `--resume` (default) and the entry gets overwritten naturally. Hand-edits make debugging future failures impossible.
4. **Redaction failures fall back to the unredacted PDF.** The paper still goes through to filter/export — but the regex anon check at the filter stage acts as the final safety net (drops papers with no "Anonymous" / "Under review" marker on page 0 or in headers). Don't disable that gate.
5. **ShareGPT entries are inference-only** — `[system, human]` turns only, no `gpt` turn. Downstream consumers tolerate `len(conversations) == 2`. If you change this, you break PaperLens's PPO-stage parsing.
6. **No hardcoded `/scratch/...` paths.** Use `--texlive-bin` (CLI flag) or `$HF_HOME` (env var). The only env var the code reads is `HF_HOME` — everything else is a flag.

---

## Critical invariants the downstream stack relies on

- **Output dir layout** is contract-stable. `paperscorer-reviewing` and the orchestrator both glob `<output_dir>/sharegpt/{text,vision}/data.json` and `<output_dir>/normalized/<id>/page_images/page_*.png`. Don't rename the dirs.
- **Anonymization regex** (filter stage) is `\banonymous\b|confidential\s+review|under\s+review|submission\s*#|blind\s+submission` (case-insensitive). PaperLens-reviewing's bug-fix history (commit `70df8cf`) tells the cautionary tale of papers using `zlab.cls` and other custom templates putting the anon marker in `\fancyfoot[L]` instead of body — the gate accepts `type=footer` blocks document-wide for that reason. Don't tighten without checking.
- **Filter mode is hard-coded to `FilterMode.ANONYMOUS`** in the normalize stage. The `core/filters.py` module supports other modes (e.g. `FilterMode.NONE`), but those exist for golden-test diffs against production, not for runtime use.
- **`<image>` token count must match `len(images[])`** in vision ShareGPT entries — every page becomes one `<image>` token in the human turn, in numerical order. Off-by-one bugs here silently produce vision rows the VL model can't parse.
- **`body_pages` is THE truncation key.** Set by the filter stage's body-pages cascade; consumed by the export stage to clip the rendered markdown AND the page_images list. Off-by-one here drops the conclusion in some cases.

---

## Sibling directories

- [`../tools/paperscorer-reviewing`](../tools/paperscorer-reviewing) — Web UI that calls `paperprep serve` (`/prepare`) for the compile + OCR + ShareGPT prep half of the score pipeline. Hard-depends on `<output_dir>/sharegpt/{text,vision}/data.json` layout + `<output_dir>/normalized/<id>/page_images/page_*.png` paths.
- [`../paperscorer-training-and-inference`](../paperscorer-training-and-inference) — Consumes the ShareGPT format paperprep emits for training (the production arxiv + iclr pipelines emit a bit-equivalent format). Golden tests in `tests/integration/test_golden_e2e_*.py` certify the equivalence.

If you change the ShareGPT schema, the page-image filename convention,
or the anonymization regex, grep both siblings before merging — they
all depend on the contract.

---

## Maintainer-only

These run on local fixtures + GPU, not in CI:

```bash
# PDF golden suite (~20 min on L40)
uv run pytest tests/integration/test_golden_e2e_pdf.py -m gpu --no-header -s

# Full LaTeX + PDF golden suite (~40 min on L40)
uv run pytest tests/integration/test_golden_e2e_latex.py -m slow --no-header -s

# Serve smoke (spins up a real paperprep serve, hits /prepare)
uv run pytest tests/integration/test_serve_smoke.py -m gpu -s
```

Set `PAPERPREP_TEXLIVE_BIN` to override the TeX Live path. Unit tests
run on every push via `.github/workflows/tests.yml`.
