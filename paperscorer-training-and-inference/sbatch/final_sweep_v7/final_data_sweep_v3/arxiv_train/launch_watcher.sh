#!/bin/bash
# Launch the auto-inference watcher for ONE arxiv-train cell.
# Runs `nohup python scripts/auto_infer_watcher.py ...` in the background;
# the watcher polls saves/.../arxiv_train/<size>/<short>/ for new
# checkpoints and submits ONE PLI inference job per checkpoint, which
# internally runs vLLM TWICE (arxiv_y24up + iclr_y25up) -- via the override
# sbatch sbatch/inference/arxiv_train_dual_eval.sbatch.
#
# Single-test mode (--test_dataset, not --test_datasets) so the watcher
# submits exactly one sbatch per checkpoint. The DATASET env var the
# watcher passes is a placeholder; the override sbatch ignores it and
# runs both hardcoded test sets.
#
# Submitted under the invoking user's account (sk7524 on login). Training
# itself runs separately on ailab from the collaborator's account.
#
# Usage:
#   bash launch_watcher.sh <cell>
#
# <cell> is one of:
#   small_text   small_vision   large_text   large_vision

set -e
cd "${SLURM_SUBMIT_DIR:-.}"
source .venv/bin/activate
mkdir -p logs/auto_inference

CELL="${1:?usage: launch_watcher.sh <small_text|small_vision|large_text|large_vision>}"

# Test pair (just used as the placeholder --test_dataset arg; real datasets
# are hardcoded inside arxiv_train_dual_eval.sbatch)
ARXIV_TEXT_Y24UP="arxiv_50_50_21k_text_wmetadata_filtered24480_y24up_test"
ARXIV_VISION_Y24UP="arxiv_50_50_21k_vision_wmetadata_filtered24480_y24up_test"

case "$CELL" in
small_text)
    SHORT=arxiv_21k_text
    TEMPLATE=qwen
    SIZE=small
    DATASET=arxiv_50_50_21k_text_wmetadata_filtered24480_train
    PLACEHOLDER_TEST=$ARXIV_TEXT_Y24UP
    IMG_ARGS=()
    ;;
small_vision)
    SHORT=arxiv_21k_vision
    TEMPLATE=qwen2_vl
    SIZE=small
    DATASET=arxiv_50_50_21k_vision_wmetadata_filtered24480_train
    PLACEHOLDER_TEST=$ARXIV_VISION_Y24UP
    IMG_ARGS=(--image_min_pixels 784 --image_max_pixels 1003520)
    ;;
large_text)
    SHORT=arxiv_balanced_per_venue_text
    TEMPLATE=qwen
    SIZE=large
    DATASET=arxiv_50_50_balanced_per_venue_text_wmetadata_train
    PLACEHOLDER_TEST=$ARXIV_TEXT_Y24UP
    IMG_ARGS=()
    ;;
large_vision)
    SHORT=arxiv_balanced_per_venue_vision
    TEMPLATE=qwen2_vl
    SIZE=large
    DATASET=arxiv_50_50_balanced_per_venue_vision_wmetadata_train
    PLACEHOLDER_TEST=$ARXIV_VISION_Y24UP
    IMG_ARGS=(--image_min_pixels 784 --image_max_pixels 1003520)
    ;;
*)
    echo "ERROR: unknown cell '$CELL'"
    echo "  must be one of: small_text small_vision large_text large_vision"
    exit 2
    ;;
esac

SAVE_DIR="saves/final_sweep_v7_datasweepv3/final_data_sweep_v3/arxiv_train/${SIZE}/${SHORT}"
RES_DIR="results/final_sweep_v7_datasweepv3/final_data_sweep_v3/arxiv_train/${SIZE}/${SHORT}"
mkdir -p "$SAVE_DIR" "$RES_DIR"

LOG_FILE="logs/auto_inference/watcher_${CELL}_$(date +%Y%m%d_%H%M%S).log"

echo "Starting watcher for cell=${CELL}"
echo "  save_dir:    ${SAVE_DIR}"
echo "  results_dir: ${RES_DIR}"
echo "  template:    ${TEMPLATE}"
echo "  override sbatch: sbatch/inference/arxiv_train_dual_eval.sbatch"
echo "    (will run both arxiv_y24up + iclr_y25up vLLM in one PLI allocation)"
echo "  log:         ${LOG_FILE}"

nohup python scripts/auto_infer_watcher.py \
    --save_dir "$SAVE_DIR" \
    --results_dir "$RES_DIR" \
    --dataset "$DATASET" \
    --test_dataset "$PLACEHOLDER_TEST" \
    --inference_sbatch "sbatch/inference/arxiv_train_dual_eval.sbatch" \
    --template "$TEMPLATE" \
    --cutoff_len 24480 \
    --max_new_tokens 1280 \
    --save_logprobs \
    --keep_safetensors_epoch_idx 2 \
    --poll_interval 60 \
    "${IMG_ARGS[@]}" \
    > "$LOG_FILE" 2>&1 &

WATCHER_PID=$!
disown $WATCHER_PID
echo "watcher started: pid=${WATCHER_PID} cell=${CELL}"
echo "tail -f $LOG_FILE   # to follow"
echo "kill ${WATCHER_PID}    # to stop"
