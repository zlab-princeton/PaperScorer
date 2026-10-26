#!/bin/bash
#SBATCH --job-name=paperlens_scaling_text_3b
#SBATCH --time=14:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gres=gpu:2
#SBATCH --partition=ailab
#SBATCH --output=logs/training/scaling_text_3b/%j.out
#SBATCH --error=logs/training/scaling_text_3b/%j.err

# Model-scaling sweep: Qwen2.5-3B text on ICLR-balanced (~21k).

cd "${SLURM_SUBMIT_DIR:-.}"
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1

SHORT_NAME="scaling_text_3b"
MODALITY="text"
TRAIN_DATASET="iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_train"
CONFIG_FILE="configs/final_sweep_v7_clean_3b.yaml"
TEMPLATE="qwen"

PER_DEVICE_BATCH=4
GRAD_ACCUM=4
NUM_GPUS=2
MASTER_PORT=29700

MODEL_DIR="saves/scaling/${SHORT_NAME}"
RESULTS_DIR="results/scaling/${SHORT_NAME}"

source sbatch/_common/canonical.sh
