# Shared canonical training pipeline for all PaperLens training sbatches.
#
# Per-sbatch files set these variables, then `source sbatch/_common/canonical.sh`:
#
#   Required:
#     SHORT_NAME, MODALITY (text|vision), TRAIN_DATASET, CONFIG_FILE, TEMPLATE,
#     MODEL_DIR, RESULTS_DIR, PER_DEVICE_BATCH, GRAD_ACCUM, NUM_GPUS, MASTER_PORT
#
#   Optional (defaults applied):
#     LR (1e-6), EPOCHS (4.0), CUTOFF_LEN (24480),
#     FSDP_CONFIG (configs/fsdp2_${NUM_GPUS}gpu_config.yaml),
#     IMG_ARGS (empty)  -- e.g. "--image_min_pixels 784 --image_max_pixels 1003520" for vision
#     EXTRA_LF_ARGS (empty)  -- extra src/train.py overrides as KEY=VAL pairs
#
# The in-job watcher fires inference on 4 cells per checkpoint:
#   arxiv_{val,test} + iclr_{val,test} -- the canonical eval matrix.

set -e

LR="${LR:-1e-6}"
EPOCHS="${EPOCHS:-4.0}"
CUTOFF_LEN="${CUTOFF_LEN:-24480}"
IMG_ARGS="${IMG_ARGS:-}"
EXTRA_LF_ARGS="${EXTRA_LF_ARGS:-}"
FSDP_CONFIG="${FSDP_CONFIG:-configs/fsdp2_${NUM_GPUS}gpu_config.yaml}"

# The canonical 4-cell eval is fixed per modality (both arxiv + iclr, val + test)
TEST_DATASETS_TEXT="\
arxiv_50_50_21k_text_wmetadata_filtered24480_y24up_test:arxiv_test,\
arxiv_50_50_21k_text_wmetadata_filtered24480_y24up_validation:arxiv_val,\
iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_y25up_test:iclr_test,\
iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_y25up_validation:iclr_val"

TEST_DATASETS_VISION="\
arxiv_50_50_21k_vision_wmetadata_filtered24480_y24up_test:arxiv_test,\
arxiv_50_50_21k_vision_wmetadata_filtered24480_y24up_validation:arxiv_val,\
iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_labelfix_v7_filtered_filtered24480_y25up_test:iclr_test,\
iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_labelfix_v7_filtered_filtered24480_y25up_validation:iclr_val"

case "${MODALITY}" in
  text)   TEST_DATASETS="${TEST_DATASETS_TEXT}";   INFERENCE_SBATCH="sbatch/inference/text_inference.sbatch" ;;
  vision) TEST_DATASETS="${TEST_DATASETS_VISION}"; INFERENCE_SBATCH="sbatch/inference/vision_inference.sbatch" ;;
  *)      echo "ERROR: MODALITY must be 'text' or 'vision' (got '${MODALITY}')"; exit 2 ;;
esac

mkdir -p "${MODEL_DIR}" "${RESULTS_DIR}"
echo "${SLURM_JOB_ID}" > "${MODEL_DIR}/.trainjob.touch"

echo "=============================================="
echo "Job:           $SLURM_JOB_ID"
echo "Short name:    ${SHORT_NAME}"
echo "Modality:      ${MODALITY}"
echo "Train dataset: ${TRAIN_DATASET}"
echo "Config:        ${CONFIG_FILE}"
echo "FSDP config:   ${FSDP_CONFIG}"
echo "LR / Epochs:   ${LR} / ${EPOCHS}"
echo "Batch:         ${NUM_GPUS} GPU x ${PER_DEVICE_BATCH} per_device x ${GRAD_ACCUM} grad_accum = $((PER_DEVICE_BATCH * NUM_GPUS * GRAD_ACCUM)) eff"
echo "Cutoff len:    ${CUTOFF_LEN}"
echo "Watcher eval:  arxiv_{test,val} + iclr_{test,val} per ckpt"
echo "Model dir:     ${MODEL_DIR}"
echo "Results dir:   ${RESULTS_DIR}"
echo "=============================================="

# ---------------------------------------------------------------
# In-job auto-inference watcher (fires 4 inference jobs per ckpt)
# ---------------------------------------------------------------
python scripts/auto_infer_watcher.py \
    --save_dir "${MODEL_DIR}" \
    --results_dir "${RESULTS_DIR}" \
    --dataset "${TRAIN_DATASET}" \
    --test_datasets "${TEST_DATASETS}" \
    --inference_sbatch "${INFERENCE_SBATCH}" \
    --template "${TEMPLATE}" \
    --cutoff_len "${CUTOFF_LEN}" \
    --max_new_tokens 1280 \
    ${IMG_ARGS} \
    --save_logprobs \
    --poll_interval 60 &
AUTO_INFER_PID=$!
echo "Auto-inference watcher started (PID: ${AUTO_INFER_PID})"

# ---------------------------------------------------------------
# Training
# ---------------------------------------------------------------
# LlamaFactory's parser resolves eval_dataset even when do_eval=false, so we
# pass the training dataset's matching _validation key as a non-failing placeholder.
EVAL_PLACEHOLDER="${EVAL_PLACEHOLDER:-${TRAIN_DATASET/_train/_validation}}"

accelerate launch \
    --config_file "${FSDP_CONFIG}" \
    --main_process_port "${MASTER_PORT}" \
    src/train.py "${CONFIG_FILE}" \
    dataset="${TRAIN_DATASET}" \
    eval_dataset="${EVAL_PLACEHOLDER}" \
    do_eval=false \
    output_dir="${MODEL_DIR}" \
    per_device_train_batch_size="${PER_DEVICE_BATCH}" \
    gradient_accumulation_steps="${GRAD_ACCUM}" \
    cutoff_len="${CUTOFF_LEN}" \
    learning_rate="${LR}" \
    num_train_epochs="${EPOCHS}" \
    lr_scheduler_type=cosine \
    'lr_scheduler_kwargs={"custom_scheduler": "cosine_then_constant", "decay_ratio": 0.75, "min_lr_rate": 0.001}' \
    ${EXTRA_LF_ARGS}

echo ""
echo "Training done. Waiting for watcher to drain..."
wait $AUTO_INFER_PID 2>/dev/null || true

echo "=============================================="
echo "COMPLETED: ${SHORT_NAME}"
echo "=============================================="
