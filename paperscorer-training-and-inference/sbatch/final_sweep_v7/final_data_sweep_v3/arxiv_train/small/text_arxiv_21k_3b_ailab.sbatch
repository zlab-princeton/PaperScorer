#!/bin/bash
#SBATCH --job-name=paperlens_arxiv21k_text_3b
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --gres=gpu:2
#SBATCH --partition=ailab
#SBATCH --output=logs/training/arxiv21k_text_3b/%j.out
#SBATCH --error=logs/training/arxiv21k_text_3b/%j.err

# Qwen2.5-3B text on arxiv-21k (main result, 3B variant).

cd "${SLURM_SUBMIT_DIR:-.}"
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1

SHORT_NAME="arxiv21k_text_3b"
MODALITY="text"
TRAIN_DATASET="arxiv_50_50_21k_text_wmetadata_filtered24480_train"
CONFIG_FILE="configs/final_sweep_v7_clean_3b.yaml"
TEMPLATE="qwen"

PER_DEVICE_BATCH=4
GRAD_ACCUM=4
NUM_GPUS=2
MASTER_PORT=29771

MODEL_DIR="saves/arxiv_train/small/${SHORT_NAME}"
RESULTS_DIR="results/arxiv_train/small/${SHORT_NAME}"

source sbatch/_common/canonical.sh
