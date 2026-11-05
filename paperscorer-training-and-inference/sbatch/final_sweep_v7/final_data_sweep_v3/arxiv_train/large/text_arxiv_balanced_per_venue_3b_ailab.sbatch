#!/bin/bash
#SBATCH --job-name=paperlens_arxivlarge_text_3b
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --gres=gpu:2
#SBATCH --partition=ailab
#SBATCH --qos=gpu-medium
#SBATCH --output=logs/training/arxivlarge_text_3b/%j.out
#SBATCH --error=logs/training/arxivlarge_text_3b/%j.err

# Qwen2.5-3B text on arxiv-balanced-per-venue (~71k papers, 3B variant).

cd "${SLURM_SUBMIT_DIR:-.}"
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1

SHORT_NAME="arxivlarge_text_3b"
MODALITY="text"
TRAIN_DATASET="arxiv_50_50_balanced_per_venue_text_wmetadata_filtered24480_train"
CONFIG_FILE="configs/final_sweep_v7_clean_3b.yaml"
TEMPLATE="qwen"

PER_DEVICE_BATCH=4
GRAD_ACCUM=4
NUM_GPUS=2
MASTER_PORT=29781

MODEL_DIR="saves/arxiv_train/large/${SHORT_NAME}"
RESULTS_DIR="results/arxiv_train/large/${SHORT_NAME}"

source sbatch/_common/canonical.sh
