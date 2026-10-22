#!/bin/bash
#SBATCH --job-name=auto_infer_text
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --partition=pli
#SBATCH --output=logs/auto_inference/%j.out
#SBATCH --error=logs/auto_inference/%j.err

# Auto-inference job for text-only models (qwen template)
# Called by checkpoint_watcher_thread.py with environment variables:
#   CHECKPOINT_DIR - Path to checkpoint directory
#   DATASET - Dataset name (with _test suffix for inference, _train for eval)
#   TEMPLATE - Template name (qwen)
#   CUTOFF_LEN - Cutoff length
#   MAX_NEW_TOKENS - Max new tokens for generation
#   RESULTS_DIR - Output directory for results
#   CKPT_STEP - Checkpoint step number
#   TRAIN_DATASET - (Optional) Training dataset name for eval metrics

set -e
cd "${SLURM_SUBMIT_DIR:-.}"
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1

# Validate required environment variables
if [ -z "$CHECKPOINT_DIR" ] || [ -z "$DATASET" ] || [ -z "$TEMPLATE" ] || [ -z "$CKPT_STEP" ] || [ -z "$RESULTS_DIR" ]; then
    echo "ERROR: Missing required environment variables"
    echo "  CHECKPOINT_DIR: ${CHECKPOINT_DIR:-<not set>}"
    echo "  DATASET: ${DATASET:-<not set>}"
    echo "  TEMPLATE: ${TEMPLATE:-<not set>}"
    echo "  CKPT_STEP: ${CKPT_STEP:-<not set>}"
    echo "  RESULTS_DIR: ${RESULTS_DIR:-<not set>}"
    exit 1
fi

# Default values
CUTOFF_LEN=${CUTOFF_LEN:-24480}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-1280}
COMPUTE_TRAIN_ACCURACY=${COMPUTE_TRAIN_ACCURACY:-0}
TRAIN_ACCURACY_MAX_SAMPLES=${TRAIN_ACCURACY_MAX_SAMPLES:-2000}
TRAIN_ACCURACY_BATCH_SIZE=${TRAIN_ACCURACY_BATCH_SIZE:-2}
DELETE_SAFETENSORS=${DELETE_SAFETENSORS:-0}
TEMPERATURE=${TEMPERATURE:-0}
SAVE_LOGPROBS=${SAVE_LOGPROBS:-0}
IS_LORA=${IS_LORA:-0}
BASE_MODEL_PATH=${BASE_MODEL_PATH:-}
LORA_RANK=${LORA_RANK:-64}

# Derive train dataset from test dataset if not provided
# e.g., dataset_test -> dataset_train
if [ -z "$TRAIN_DATASET" ]; then
    TRAIN_DATASET="${DATASET%_test}_train"
fi

echo "=============================================="
echo "Auto-Inference Job (Text)"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Checkpoint: ${CHECKPOINT_DIR}"
echo "Test Dataset: ${DATASET}"
echo "Train Dataset: ${TRAIN_DATASET}"
echo "Template: ${TEMPLATE}"
echo "Step: ${CKPT_STEP}"
echo "Results Dir: ${RESULTS_DIR}"
echo "Cutoff Len: ${CUTOFF_LEN}"
echo "Max New Tokens: ${MAX_NEW_TOKENS}"
echo "Compute Train Accuracy: ${COMPUTE_TRAIN_ACCURACY}"
echo "Train Accuracy Max Samples: ${TRAIN_ACCURACY_MAX_SAMPLES}"
echo "Train Accuracy Batch Size: ${TRAIN_ACCURACY_BATCH_SIZE}"
echo "Delete Safetensors: ${DELETE_SAFETENSORS}"
echo "Temperature: ${TEMPERATURE}"
echo "Save Logprobs: ${SAVE_LOGPROBS}"
echo "Is LoRA: ${IS_LORA}"
echo "Base Model Path: ${BASE_MODEL_PATH:-<not set>}"
echo "=============================================="

# Compute model/adapter paths for LoRA vs full FT
MODEL_PATH="${CHECKPOINT_DIR}"
ADAPTER_ARG=""
LORA_VLLM_CONFIG=""
if [ "$IS_LORA" = "1" ]; then
    MODEL_PATH="${BASE_MODEL_PATH}"
    ADAPTER_ARG="--adapter_name_or_path ${CHECKPOINT_DIR}"
    LORA_VLLM_CONFIG="--vllm_config {\"max_lora_rank\": ${LORA_RANK}}"
    echo "LoRA mode: base=${MODEL_PATH}, adapter=${CHECKPOINT_DIR}, rank=${LORA_RANK}"
fi

# Create directories
mkdir -p logs/auto_inference "${RESULTS_DIR}"

# Output file names
TRAIN_METRICS_FILE="${RESULTS_DIR}/train-ckpt-${CKPT_STEP}.json"
OUTPUT_FILE="${RESULTS_DIR}/finetuned-ckpt-${CKPT_STEP}.jsonl"

# ============================================
# Step 1: Eval on training data for metrics (optional)
# ============================================
if [ "$COMPUTE_TRAIN_ACCURACY" = "1" ]; then
    echo ""
    echo "=== Step 1: Eval on training data ==="
    echo "Output: ${TRAIN_METRICS_FILE}"
    echo ""

    python scripts/eval_training_ckpt.py \
        --model_name_or_path "${MODEL_PATH}" \
        ${ADAPTER_ARG} \
        --dataset "${TRAIN_DATASET}" \
        --template "${TEMPLATE}" \
        --cutoff_len ${CUTOFF_LEN} \
        --save_name "${TRAIN_METRICS_FILE}" \
        --sft_accuracy_format boxed \
        --sft_positive_token Accept \
        --sft_negative_token Reject \
        --per_device_eval_batch_size ${TRAIN_ACCURACY_BATCH_SIZE} \
        --max_samples ${TRAIN_ACCURACY_MAX_SAMPLES} \
        --test_dataset "${DATASET}"

    echo ""
    echo "=== Step 1 Complete ==="
    echo ""
else
    echo ""
    echo "=== Step 1: Skipped (COMPUTE_TRAIN_ACCURACY=0) ==="
    echo ""
fi

# ============================================
# Step 2: vLLM inference on test data
# ============================================
echo ""
echo "=== Step 2: vLLM inference on test data ==="
echo "Output: ${OUTPUT_FILE}"
echo ""

# Build extra args for temperature and logprobs
INFER_EXTRA_ARGS=""
if [ "$TEMPERATURE" != "" ]; then
    INFER_EXTRA_ARGS="$INFER_EXTRA_ARGS --temperature $TEMPERATURE"
fi
if [ "$SAVE_LOGPROBS" = "1" ]; then
    INFER_EXTRA_ARGS="$INFER_EXTRA_ARGS --save_logprobs"
fi

# Run inference (no image params for text-only)
python scripts/vllm_infer.py \
    --model_name_or_path "${MODEL_PATH}" \
    ${ADAPTER_ARG} \
    --dataset "${DATASET}" \
    --template "${TEMPLATE}" \
    --cutoff_len "${CUTOFF_LEN}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    ${LORA_VLLM_CONFIG} \
    ${INFER_EXTRA_ARGS} \
    --save_name "${OUTPUT_FILE}"

# Write .inferdone.touch file to signal completion
echo "${OUTPUT_FILE}" > "${CHECKPOINT_DIR}/.inferdone.touch"

# ============================================
# Step 3: Delete safetensors to save disk space (optional)
# ============================================
if [ "$DELETE_SAFETENSORS" = "1" ] && [ "$IS_LORA" != "1" ]; then
    echo ""
    echo "=== Step 3: Cleaning up safetensors ==="
    SAFETENSOR_COUNT=$(ls "${CHECKPOINT_DIR}"/*.safetensors 2>/dev/null | wc -l)
    if [ "$SAFETENSOR_COUNT" -gt 0 ]; then
        rm -f "${CHECKPOINT_DIR}"/*.safetensors
        touch "${CHECKPOINT_DIR}/.cleaned.touch"
        echo "Cleaned ${SAFETENSOR_COUNT} safetensors file(s) from ${CHECKPOINT_DIR}"
    else
        echo "No safetensors files found in ${CHECKPOINT_DIR}"
    fi
elif [ "$IS_LORA" = "1" ]; then
    echo ""
    echo "=== Step 3: Skipped (LoRA adapters are small, keeping them) ==="
fi

echo ""
echo "=============================================="
echo "COMPLETED: checkpoint-${CKPT_STEP}"
echo "Train metrics saved to: ${TRAIN_METRICS_FILE}"
echo "Test predictions saved to: ${OUTPUT_FILE}"
echo "=============================================="
