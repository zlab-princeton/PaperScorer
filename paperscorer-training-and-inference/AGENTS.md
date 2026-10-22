# Agent / LLM Guide — paperlens-training-and-inference

This file is the entry point for any agent (Claude Code, Codex, Cursor, etc.)
landing in the repo. The README is for humans replicating experiments; this
file is for agents navigating the code. CLAUDE.md is a symlink to this file —
both CLIs pick up the same content from the cwd.

---

## What this repo is

Training + inference code for **PaperLens** — SFT models that predict paper
acceptance from anonymized text or rendered page images. Forked from
LLaMA-Factory; most of `src/` is upstream.

The two human-facing docs:
- [`README.md`](README.md) — install, env vars, replication quickstart.
- [`EXPERIMENTS.md`](EXPERIMENTS.md) — each paper result mapped to its sbatch + figure script.

---

## Where to start a task

| Task | Start here |
|---|---|
| Reproduce a paper result | `EXPERIMENTS.md` → find the category → run the listed sbatch / script. |
| Materialize a dataset | `scripts/reconstruction.py` (HF release → local `data/<key>/data.json`). |
| Add a new training run | Copy a sbatch in `sbatch/final_sweep_v7/.../{small,large,scaling}/`. Each sources `sbatch/_common/canonical.sh` — only the per-run vars change. |
| Score an existing ckpt offline | `sbatch sbatch/inference/{text,vision}_inference.sbatch <ckpt> <dataset_key>` (or use `scripts/auto_infer_watcher.py` in-job). |
| Fit calibration / produce figures | `scripts/calibration_posthoc.py` → `scripts/figures/generate_calibration*.py`. |
| Frontier-model API baseline | See `EXPERIMENTS.md` §5. Requires `PORTKEY_API_KEY` and `pip install portkey-ai`. Pattern lives in `scripts/frontier_memory_probe.py`. The full-test runner (`scripts/api_baseline_infer.py`) is **TODO**. |

---

## Hard rules

1. **Do not refactor `src/llamafactory/`.** It's upstream LLaMA-Factory; small patches OK, rewrites are not.
2. **Every training sbatch's watcher must hit all 4 eval cells** (`arxiv_{test,val}` + `iclr_{test,val}`). The figure scripts assume `results/<run>/<cell>/finetuned-ckpt-<step>.jsonl` exists for all four; if you skip a cell you'll break figures silently.
3. **Reconstruction round-trip is byte-identical.** The CI smoke test (`.github/workflows/reconstruction_smoke.yml`) catches obvious regressions. The full byte-identical round-trip is maintainer-only (run against a local `hf_release_v2/` mirror with `PAPERLENS_HF_LOCAL_DIR` set).
4. **HF cache + vLLM cache must live on scratch, not `$HOME`.** Set `HF_HOME` and `VLLM_CACHE_ROOT` before launching slurm jobs — see [Environment](README.md#environment) in the README.
5. **No q4 stratified subsetting anywhere.** The legacy q4 paths (subsample_dataset.py, agent_q4) were deleted; the API + agent baselines run on the **full** `*_test` set.
6. **Don't hardcode `/scratch/...` paths.** Sbatches use `cd "${SLURM_SUBMIT_DIR:-.}"`; scripts resolve the repo root via `Path(__file__).resolve().parents[1]`. Anything user-specific belongs behind an env var.

---

## Critical invariants the codebase relies on

- **Canonical 4-cell eval string** (see `sbatch/_common/canonical.sh`): every train sbatch passes the exact same `TEST_DATASETS_TEXT` / `TEST_DATASETS_VISION` string to the watcher. Figure scripts hardcode `arxiv_test`, `arxiv_val`, `iclr_test`, `iclr_val` as the cell directory names.
- **Logprob extraction during inference**: the SFT vision model emits logprobs at the decision position. `scripts/vllm_infer.py` writes `logp_accept` + `logp_reject` per row; downstream calibration (`scripts/calibration_posthoc.py`) treats `z = logp_accept - logp_reject` as the raw decision logit. Don't drop logprobs from the inference output.
- **Platt scaling parameters** live in `scripts/calibration_posthoc.py` and are fit per (domain, modality). Four (a, b) rows: `(iclr,text)`, `(iclr,vision)`, `(arxiv,text)`, `(arxiv,vision)`. The sibling `tools/paperlens-reviewing` and `tools/paperlens-arxiv-server` reuse the same numbers — keep them in sync if you re-fit.
- **Released model checkpoints are immutable**: `arxiv_21k_vision_3b/checkpoint-5236` (3B) and `arxiv_21k_vision/checkpoint-2618` (7B) are the canonical ckpts referenced by every downstream tool. Don't rename or overwrite.

---

## Sibling repos

- [`../tools/paperlens-reviewing`](../tools/paperlens-reviewing) — interactive UI: drop in a PDF / LaTeX dir / arXiv link, get a calibrated decision plus optional Claude Code / Codex agentic review. Reuses our Platt params and the served checkpoint.
- [`../tools/paperlens-arxiv-server`](../tools/paperlens-arxiv-server) — paperlens-as-reranker stacked on top of arxiv_retriever; accept-prob-ranked literature search.

If you touch the Platt params, the sharegpt schema, or the canonical eval cell names, you'll affect both siblings. Grep both before changing.
