#!/bin/bash
#SBATCH --job-name=paperlens_dscaling_vision_7b
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --gres=gpu:4
#SBATCH --partition=ailab
#SBATCH --array=0-1
#SBATCH --output=logs/training/dscaling_vision_7b/%A_%a.out
#SBATCH --error=logs/training/dscaling_vision_7b/%A_%a.err

# Data-scaling sweep: Qwen2.5-VL-7B vision on {50,75}% ICLR subsets.

cd "${SLURM_SUBMIT_DIR:-.}"
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1

FRACTIONS=("50pct" "75pct")
SUBSET="${FRACTIONS[$SLURM_ARRAY_TASK_ID]}"

SHORT_NAME="dscaling_vision_7b_${SUBSET}"
MODALITY="vision"
BASE_DATASET="iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_labelfix_v7_filtered_filtered24480"
TRAIN_DATASET="${BASE_DATASET}_train_${SUBSET}"
EVAL_PLACEHOLDER="${BASE_DATASET}_validation"
CONFIG_FILE="configs/final_sweep_v7_vision.yaml"
TEMPLATE="qwen2_vl"

PER_DEVICE_BATCH=4
GRAD_ACCUM=1
NUM_GPUS=4
MASTER_PORT=$((29726 + SLURM_ARRAY_TASK_ID))

IMG_ARGS="--image_min_pixels 784 --image_max_pixels 1003520"

MODEL_DIR="saves/data_scaling/${SHORT_NAME}"
RESULTS_DIR="results/data_scaling/${SHORT_NAME}"

source sbatch/_common/canonical.sh
