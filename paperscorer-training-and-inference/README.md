# PaperLens — Training & Inference

Reproducible code + sbatches for **PaperLens**, a suite of SFT models that
predict paper-acceptance outcomes from full paper bodies (text) or rendered
page images (vision). Built on top of [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).

- **8 released models** (4 sizes × 2 modalities, both arxiv- and iclr-trained):
  → [PaperScorer collection on HuggingFace](https://huggingface.co/collections/zlab-princeton/paperscorer)
- **Released datasets** (4 splits per modality + domain):
  - [skonan/PaperScorer-Text](https://huggingface.co/datasets/skonan/PaperScorer-Text) — subsets `arxiv` + `openreview-iclr`
  - [skonan/PaperScorer-Vision](https://huggingface.co/datasets/skonan/PaperScorer-Vision) — subsets `arxiv` + `openreview-iclr` (image-bytes embedded)
- **Sibling tools**:
  - [paperlens-reviewing](../tools/paperlens-reviewing) — interactive UI for scoring a single paper (PDF / LaTeX dir / arXiv link) with optional agentic-review pass.
  - [paperlens-arxiv-server](../tools/paperlens-arxiv-server) — stacks the PaperLens reranker on top of arxiv_retriever for accept-prob-ranked search.

See [`EXPERIMENTS.md`](EXPERIMENTS.md) for the experiment → sbatch → figure-script
mapping. Agent-friendly pointer files at the repo root: [`AGENTS.md`](AGENTS.md) and [`CLAUDE.md`](CLAUDE.md).

---

## Install

```bash
git clone https://github.com/<TODO>/paperlens-training-and-inference
cd paperlens-training-and-inference
uv venv .venv && source .venv/bin/activate
uv pip install -e ".[torch,metrics,deepspeed,liger-kernel]"
```

System deps: CUDA-capable GPU (vLLM + FSDP2 require A100 / H100 / H200), TeX
Live (only for re-rendering paper figures via `scripts/figures/`).

For the **PortKey API baselines** (frontier-model comparison rows in the main
results table) install the optional extra:
```bash
uv pip install portkey-ai
```

### One-time warmup: download base models + released PaperLens checkpoints

The training sbatches set `TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1` for speed
and reproducibility — which means the model bytes must already sit in your HF
cache (`$HF_HOME`). Do this once before the first slurm submit:

```bash
# Base Qwen models — pick the sizes you'll train
uv run hf download Qwen/Qwen2.5-7B-Instruct
uv run hf download Qwen/Qwen2.5-VL-7B-Instruct
# 3B / 14B / 32B / VL-3B / VL-32B as needed
uv run hf download Qwen/Qwen2.5-3B-Instruct
uv run hf download Qwen/Qwen2.5-VL-3B-Instruct

# Released PaperLens SFT models (skip if you're training from scratch)
# Full collection: https://huggingface.co/collections/zlab-princeton/paperscorer
uv run hf download skonan/paperscorer-7b-vision-arxiv     # 7B vision, arxiv-trained (default in configs/serve.yaml)
uv run hf download skonan/paperscorer-3b-vision-arxiv     # 3B vision (lower memory)
uv run hf download skonan/paperscorer-7b-text-arxiv     # 7B text
# ...etc — 8 variants total (T/V × 3B/7B × arxiv/iclr)
```

For inference + the reviewing/arxiv-server sibling tools, the **published
PaperLens models are the default** — no local SFT checkpoint required.
Point `configs/serve.yaml:ckpt_path` at any of the released repo ids above
(it defaults to `skonan/paperscorer-7b-vision-arxiv`).

---

## Environment

The training + inference scripts read these env vars; export them in your
shell or your slurm sbatch before launching anything. Required vs optional
is called out per variable.

| Var | Required for | Purpose |
|---|---|---|
| `HF_HOME` | every script that loads HF models or datasets | HuggingFace cache root. Defaults to `~/.cache/huggingface` — override on slurm to point at a shared scratch path so dataset/model bytes are cached once, not per-job. |
| `HF_HUB_OFFLINE=1` | optional | Skip network hits on hub after first download. Every shipped sbatch sets this. |
| `TRANSFORMERS_OFFLINE=1` | optional | Same idea for the transformers library's HF lookups. Every shipped sbatch sets this. |
| `VLLM_CACHE_ROOT` | vLLM inference (`scripts/vllm_infer.py`, `scripts/run_inference.py`) | Where vLLM caches compiled engine artifacts. Point at scratch on slurm. |
| `WANDB_API_KEY` | training (optional) | If set, `accelerate launch` logs to wandb. `WANDB_DISABLED=1` to silence. |
| `PORTKEY_API_KEY` | PortKey API baselines (`scripts/frontier_memory_probe.py`, future `api_baseline_infer.py`) | Auth for the PortKey gateway that routes to Claude / GPT-5.4 / Gemini-3.1-Pro. |
| `GEMINI_API_KEY` | Vertex AI batch path (`scripts/gemini_batch_*.py`) | Alternative cheaper batch path for Gemini-only runs (50% of online cost). PortKey is the canonical multi-provider path. |
| `PAPERLENS_HF_LOCAL_DIR` | `scripts/reconstruction.py` (optional) | Override the HF cache resolution to read released datasets from a local mirror dir. Useful on slurm with restricted egress. |
| `PAPERLENS_TEST_CKPT_PATH` | `scripts/tests/test_serve_idempotency.py` (optional) | Local checkpoint path the serve-idempotency test loads. Skips the test when unset. |

---

## Quick start: reconstruct the training data

The released HF datasets are content stores (one row per unique paper). To
materialize the LLaMA-Factory-formatted local data (`data/<key>/data.json` +
image PNGs) for one or more dataset_info entries, run:

```bash
# Canonical 4-cell eval datasets (small + fast)
python scripts/reconstruction.py --dataset_keys \
    arxiv_50_50_21k_text_wmetadata_filtered24480_y24up_test \
    arxiv_50_50_21k_text_wmetadata_filtered24480_y24up_validation \
    iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_y25up_test \
    iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_y25up_validation

# Or rebuild every publishable key (large — pulls all vision image bytes)
python scripts/reconstruction.py --all --local_dir /path/to/hf_release_local
```

Round-trip is **byte-identical** for conv[1] (system + user + assistant),
`_metadata`, and (for vision) image dimensions. Verified by
`scripts/tests/test_reconstruction.py`.

For **panel** datasets (text + a single page-grid image, used in cross-modality
ablations) the HF release **does not** ship them — they're rebuilt locally:

```bash
# After reconstruction.py has materialized the base vision data:
python scripts/build_panel_images.py --workers 32
python scripts/build_panel_dataset.py        # vision-panel rows
python scripts/build_text_panel_dataset.py   # text-with-panel rows
```

---

## Cluster configuration

The shipped sbatches are wired for **Princeton della** — `ailab` for training,
`pli` for the auto-fired inference jobs. **Edit the `#SBATCH --partition` /
time / mem / `--gres=gpu:N` lines (add `--account=…` / `--qos=…` if your
cluster requires them) before submitting anything.**

What the headers look like today:

```bash
# sbatch/final_sweep_v7/.../arxiv_train/small/text_arxiv_21k_7b_ailab.sbatch  (training)
#!/bin/bash
#SBATCH --job-name=paperlens_arxiv21k_text_7b
#SBATCH --time=14:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gres=gpu:4
#SBATCH --partition=ailab                  # ← change for your training partition
#SBATCH --output=logs/training/arxiv21k_text_7b/%j.out
#SBATCH --error=logs/training/arxiv21k_text_7b/%j.err
```

```bash
# sbatch/inference/text_inference.sbatch  (auto-fired by the in-job watcher)
#!/bin/bash
#SBATCH --job-name=auto_infer_text
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --partition=pli                    # ← change for your inference partition
#SBATCH --output=logs/auto_inference/%j.out
#SBATCH --error=logs/auto_inference/%j.err
```

---

## Quick start: training

All training sbatches share a canonical template — set a few vars, source the
common header. Example: train Qwen2.5-7B (text) on arxiv-21k:

```bash
sbatch sbatch/final_sweep_v7/final_data_sweep_v3/arxiv_train/small/text_arxiv_21k_7b_ailab.sbatch
```

The in-job watcher (`scripts/auto_infer_watcher.py`) fires inference on
4 cells per checkpoint (using the inference sbatch above):

- `arxiv_50_50_21k_*_y24up_test` (in-distribution test, arxiv)
- `arxiv_50_50_21k_*_y24up_validation` (in-distribution val, arxiv)
- `iclr_*_y25up_test` (cross-domain test, iclr)
- `iclr_*_y25up_validation` (cross-domain val, iclr)

Outputs land at `results/<short_name>/{arxiv_test,arxiv_val,iclr_test,iclr_val}/finetuned-ckpt-<step>.jsonl`.

See [`sbatch/_common/canonical.sh`](sbatch/_common/canonical.sh) for the shared
training pipeline; per-sbatch files are ~25 lines and only set per-run vars.

---

## Quick start: inference + calibration

```bash
# Score a single ckpt × dataset (manual; watcher does this automatically)
sbatch sbatch/inference/text_inference.sbatch <ckpt_dir> <dataset_key>

# Fit + apply validation-set calibration to test predictions
python scripts/calibration_posthoc.py --val_jsonl <val.jsonl> --test_jsonl <test.jsonl>

# In-distribution + cross-domain accuracy with calibrated thresholds
python scripts/iclr_calibrated_acc_2526.py --results_root results/
python scripts/arxiv_calibrated_acc.py --results_root results/

# Generate paper figures
python scripts/figures/generate_calibration.py --result_root results/
```

---

## Project layout

```
paperlens-training-and-inference/
├── src/                       # LLaMA-Factory source (mostly upstream + a few patches)
├── sbatch/
│   ├── _common/canonical.sh   # Shared training pipeline (every train sbatch sources this)
│   ├── final_sweep_v7/        # Training sbatches (small/, large/, scaling/, data_scaling/)
│   └── inference/             # text_inference.sbatch, vision_inference.sbatch (single-GPU; edit partition for your cluster)
├── scripts/
│   ├── reconstruction.py             # HF release -> local data/<key>/data.json
│   ├── build_panel_*.py              # Rebuild panel views locally from base data
│   ├── auto_infer_watcher.py         # In-job watcher (used by every train sbatch)
│   ├── vllm_infer.py, eval_training_ckpt.py
│   ├── calibration_posthoc.py
│   ├── iclr_calibrated_acc_2526.py, arxiv_calibrated_acc.py
│   ├── gemini_batch_*.py             # Vertex AI batch path (Gemini-only, 50% cost)
│   ├── frontier_memory_probe.py      # PortKey-gateway memory-probe experiment
│   ├── figures/                      # Paper figure generators (matplotlib + scipy)
│   ├── tests/test_reconstruction.py  # Round-trip integrity test
│   └── stat_utils/, convert_ckpt/, api_example/  # LLaMA-Factory stock helpers
├── configs/
│   ├── final_sweep_v7_clean[_3b|_14b|_32b|_32b_lora{,_r128}].yaml  # Text 3B/7B/14B/32B
│   ├── final_sweep_v7_vision[_3b|_32b|_32b_lora{,_r128}].yaml      # Vision VL variants
│   └── fsdp2_{1,2,4,8}gpu_config.yaml                              # Accelerate FSDP2
├── AGENTS.md / CLAUDE.md      # LLM/agent pointer files for repo navigation
├── README.md                  # this file
└── EXPERIMENTS.md             # paper-result → sbatch → figure mapping
```

---

## Citation

```bibtex
@inproceedings{konan2026paperscorer,
  title     = {PaperScorer: How Predictable Is Paper Acceptance?},
  author    = {Konan, Sachin and Liu, Jonathan and Liu, Zhuang},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

Upstream LLaMA-Factory: [Zheng et al., ACL 2024](https://aclanthology.org/2024.acl-demos.38/).
