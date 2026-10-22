# Experiments

This document maps each paper-result category to (a) the dataset reconstruction
calls, (b) the sbatches that produce model checkpoints + eval JSONLs, and
(c) the figure scripts that consume the results.

Every training sbatch's in-job watcher fires inference on the canonical **4-cell
eval** per ckpt: `arxiv_{test, val}` + `iclr_{test, val}`. Results land at
`results/<short_name>/<cell>/finetuned-ckpt-<step>.jsonl`.

---

## 1. Main results (3B + 7B × text + vision × arxiv-21k + arxiv-balanced-71k)

The headline numbers. SFT trained on either arxiv-21k or arxiv-balanced-per-venue
(~71k), evaluated zero-shot on iclr (cross-domain) and in-distribution on arxiv.

**Reconstruct training data**:
```bash
python scripts/reconstruction.py --dataset_keys \
    arxiv_50_50_21k_text_wmetadata_filtered24480_train \
    arxiv_50_50_21k_vision_wmetadata_filtered24480_train \
    arxiv_50_50_balanced_per_venue_text_wmetadata_filtered24480_train \
    arxiv_50_50_balanced_per_venue_vision_wmetadata_filtered24480_train
```

**Training sbatches** (8 total):

| Modality | Size | Dataset (arxiv-21k → small/, balanced-71k → large/) |
|---|---|---|
| text | 7B | `sbatch/.../arxiv_train/small/text_arxiv_21k_7b_ailab.sbatch` |
| text | 3B | `sbatch/.../arxiv_train/small/text_arxiv_21k_3b_ailab.sbatch` |
| vision | 7B | `sbatch/.../arxiv_train/small/vision_arxiv_21k_7b_ailab.sbatch` |
| vision | 3B | `sbatch/.../arxiv_train/small/vision_arxiv_21k_3b_ailab.sbatch` |
| text | 7B | `sbatch/.../arxiv_train/large/text_arxiv_balanced_per_venue_7b_ailab.sbatch` |
| text | 3B | `sbatch/.../arxiv_train/large/text_arxiv_balanced_per_venue_3b_ailab.sbatch` |
| vision | 7B | `sbatch/.../arxiv_train/large/vision_arxiv_balanced_per_venue_7b_ailab.sbatch` |
| vision | 3B | `sbatch/.../arxiv_train/large/vision_arxiv_balanced_per_venue_3b_ailab.sbatch` |

**Figures**:
- `scripts/figures/generate_3b_vs_7b.py` — model size vs accuracy
- `scripts/figures/generate_7b_vision_arxiv_small_vs_large.py` — dataset size effect

---

## 2. Calibration experiments

Validation-fit calibrated thresholds applied to test logit gaps. No separate
training — uses the eval JSONLs from category 1 + 3.

**Scripts**:
- `scripts/calibration_posthoc.py` — fits per-domain Platt scaling on val logits
- `scripts/iclr_calibrated_acc_2526.py` — applies calibrated thresholds to iclr test
- `scripts/arxiv_calibrated_acc.py` — applies calibrated thresholds to arxiv test

**Figures** (`scripts/figures/`):
- `generate_calibration.py`, `generate_calibration_1x2.py`, `generate_calibration_1x3.py`
- `generate_calibration_extended.py`, `generate_calibration_extended_2025.py`
- `generate_calibration_platt.py`, `generate_calibration_posthoc.py`
- `generate_calibration_val_correction.py` — see also category 5
- `generate_calibration_text_vs_vision_wd.py`
- `generate_calibration_ratio_sweep.py`

---

## 3. Model-scaling experiments (3B → 14B → 32B + LoRA on ICLR)

Train sweep at sizes {3B, 14B, 32B (full), 32B LoRA r=64, 32B LoRA r=128} × {text, vision}.

**Reconstruct training data**:
```bash
python scripts/reconstruction.py --dataset_keys \
    iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_train \
    iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_labelfix_v7_filtered_filtered24480_train
```

**Training sbatches** (9 in `sbatch/.../optim_search_2026/scaling/`):
- `text_3b_ailab.sbatch`, `text_14b_ailab.sbatch`, `text_32b_ailab.sbatch`,
  `text_32b_lora_ailab.sbatch`, `text_32b_lora_r128_ailab.sbatch`
- `vision_3b_ailab.sbatch`, `vision_32b_ailab.sbatch`,
  `vision_32b_lora_ailab.sbatch`, `vision_32b_lora_r128_ailab.sbatch`

**Figures**:
- `scripts/figures/generate_scaling_*.py`

---

## 4. Data-scaling experiments

Subset sweep at {50%, 75%, 100%} of the iclr training data. Slurm job arrays
fire one job per fraction.

**Reconstruct training data**:
```bash
python scripts/reconstruction.py --dataset_keys \
    iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_train_50pct \
    iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_train_75pct \
    iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_labelfix_v7_filtered_filtered24480_train_50pct \
    iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_labelfix_v7_filtered_filtered24480_train_75pct
```

**Training sbatches** (4 in `sbatch/.../optim_search_2026/data_scaling/`, each a slurm-array):
- `text_3b_subsets_ailab.sbatch`, `text_7b_subsets_ailab.sbatch`
- `vision_3b_subsets_ailab.sbatch`, `vision_7b_subsets_ailab.sbatch`

The 100% baseline comes from category 3's full ICLR runs.

**Figures**:
- `scripts/figures/generate_scaling_laws.py` (data axis)

---

## 5. PortKey frontier-model API baselines

LLM-judge baselines for the main results table. Each frontier model receives
the same single-shot `\boxed{rating, decision}` prompt over the full standard
`*_test` set, routed through the **PortKey gateway** so Claude / GPT-5.4 /
Gemini-3.1-Pro all share one transport + one set of retry semantics.

**Dependency**: `pip install portkey-ai`, then `export PORTKEY_API_KEY=...`
(see [Environment](README.md#environment) for the full env-var table).
The PortKey pattern lives in `scripts/frontier_memory_probe.py` (a sibling
memory-probe experiment); reuse `call_portkey(client, model, prompt,
max_tokens)` and `reasoning_kwargs(model)` from there.

**Reconstruct test data**:
```bash
python scripts/reconstruction.py --dataset_keys \
    arxiv_50_50_21k_text_wmetadata_filtered24480_y24up_test \
    arxiv_50_50_21k_vision_wmetadata_filtered24480_y24up_test \
    iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_y25up_test \
    iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_labelfix_v7_filtered_filtered24480_y25up_test
```

**Runner**:
- `scripts/api_baseline_infer.py` (**TODO** — not yet in tree). Should emit
  one jsonl row per `(model, dataset_key, paper_id)` with fields matching
  the local SFT jsonls (`pred`, `gold`, `logp_accept`, `logp_reject`) so the
  same calibration + accuracy scripts work unchanged. Reuse the parsing
  helpers from `scripts/grade_agent_predictions.py` for `\boxed{}` parsing.

**Cheap Gemini-only alternative**:
- `scripts/gemini_batch_submit.py` / `gemini_batch_retrieve.py` /
  `gemini_batch_review.py` — Vertex AI batch prediction with GCS image URIs
  (50% cost savings vs online). Doesn't go through PortKey, doesn't cover
  Claude or GPT — kept for the Gemini-specific cross-check rows.

**Grading + figures**:
- `scripts/grade_agent_predictions.py` — parse + score `\boxed{}` outputs vs gold labels
- `scripts/figures/generate_calibration.py` and the rest of category 2 read
  the resulting jsonls the same way they read SFT jsonls.

---

## 6. Reverse-label ablation (label flip robustness)

(Out of the 5 main categories but datasets are released for completeness.)

Training keys with `_reversed` suffix have the gold labels inverted. Useful
for measuring how much of the SFT signal is genuine vs. label memorization.

Datasets:
- `iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_{train,test,validation}_reversed`
- Same for vision

No published sbatch — to run: copy a scaling sbatch and override `TRAIN_DATASET` to a `_reversed` variant.

---

## Quick reference: canonical 4-cell eval

Every training sbatch's in-job watcher passes this `TEST_DATASETS` string (for
text — the vision variant is in `sbatch/_common/canonical.sh`):

```
arxiv_50_50_21k_text_wmetadata_filtered24480_y24up_test:arxiv_test,
arxiv_50_50_21k_text_wmetadata_filtered24480_y24up_validation:arxiv_val,
iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_y25up_test:iclr_test,
iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_y25up_validation:iclr_val
```

So `results/<run>/arxiv_test/finetuned-ckpt-<step>.jsonl` etc. always exist
for every training run; figure scripts can assume this layout.
