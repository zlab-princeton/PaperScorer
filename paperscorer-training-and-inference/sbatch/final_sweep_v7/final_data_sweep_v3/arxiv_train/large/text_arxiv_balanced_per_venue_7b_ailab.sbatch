#!/bin/bash
#SBATCH --job-name=paperlens_arxivlarge_text_7b
#SBATCH --time=32:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gres=gpu:4
#SBATCH --partition=ailab
#SBATCH --qos=gpu-medium
#SBATCH --output=logs/training/arxivlarge_text_7b/%j.out
#SBATCH --error=logs/training/arxivlarge_text_7b/%j.err

# Qwen2.5-7B text on arxiv-balanced-per-venue (~71k papers, main result).

cd "${SLURM_SUBMIT_DIR:-.}"
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1

SHORT_NAME="arxivlarge_text_7b"
MODALITY="text"
TRAIN_DATASET="arxiv_50_50_balanced_per_venue_text_wmetadata_filtered24480_train"
CONFIG_FILE="configs/final_sweep_v7_clean.yaml"
TEMPLATE="qwen"

PER_DEVICE_BATCH=4
GRAD_ACCUM=2
NUM_GPUS=4
MASTER_PORT=29780

MODEL_DIR="saves/arxiv_train/large/${SHORT_NAME}"
RESULTS_DIR="results/arxiv_train/large/${SHORT_NAME}"

source sbatch/_common/canonical.sh
