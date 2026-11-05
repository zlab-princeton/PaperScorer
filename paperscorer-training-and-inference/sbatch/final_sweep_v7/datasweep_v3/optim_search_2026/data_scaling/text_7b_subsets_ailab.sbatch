#!/bin/bash
#SBATCH --job-name=paperlens_dscaling_text_7b
#SBATCH --time=14:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gres=gpu:4
#SBATCH --partition=ailab
#SBATCH --array=0-1
#SBATCH --output=logs/training/dscaling_text_7b/%A_%a.out
#SBATCH --error=logs/training/dscaling_text_7b/%A_%a.err

# Data-scaling sweep: Qwen2.5-7B text on {50,75}% ICLR subsets.

cd "${SLURM_SUBMIT_DIR:-.}"
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1

FRACTIONS=("50pct" "75pct")
SUBSET="${FRACTIONS[$SLURM_ARRAY_TASK_ID]}"

SHORT_NAME="dscaling_text_7b_${SUBSET}"
MODALITY="text"
BASE_DATASET="iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered"
TRAIN_DATASET="${BASE_DATASET}_train_${SUBSET}"
EVAL_PLACEHOLDER="${BASE_DATASET}_validation"
CONFIG_FILE="configs/final_sweep_v7_clean.yaml"
TEMPLATE="qwen"

PER_DEVICE_BATCH=4
GRAD_ACCUM=2
NUM_GPUS=4
MASTER_PORT=$((29722 + SLURM_ARRAY_TASK_ID))

MODEL_DIR="saves/data_scaling/${SHORT_NAME}"
RESULTS_DIR="results/data_scaling/${SHORT_NAME}"

source sbatch/_common/canonical.sh
