#!/bin/bash
#SBATCH --job-name=paperlens_arxiv21k_text_7b
#SBATCH --time=14:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gres=gpu:4
#SBATCH --partition=ailab
#SBATCH --output=logs/training/arxiv21k_text_7b/%j.out
#SBATCH --error=logs/training/arxiv21k_text_7b/%j.err

# Qwen2.5-7B text on arxiv-21k (main result). In-job watcher fires 4-cell
# eval (arxiv+iclr x val+test) per checkpoint.

cd "${SLURM_SUBMIT_DIR:-.}"
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1

SHORT_NAME="arxiv21k_text_7b"
MODALITY="text"
TRAIN_DATASET="arxiv_50_50_21k_text_wmetadata_filtered24480_train"
CONFIG_FILE="configs/final_sweep_v7_clean.yaml"
TEMPLATE="qwen"

PER_DEVICE_BATCH=4
GRAD_ACCUM=2
NUM_GPUS=4
MASTER_PORT=29770

MODEL_DIR="saves/arxiv_train/small/${SHORT_NAME}"
RESULTS_DIR="results/arxiv_train/small/${SHORT_NAME}"

source sbatch/_common/canonical.sh
