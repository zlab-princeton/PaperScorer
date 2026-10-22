#!/bin/bash
#SBATCH --job-name=paperlens_arxiv21k_vision_3b
#SBATCH --time=14:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --gres=gpu:2
#SBATCH --partition=ailab
#SBATCH --output=logs/training/arxiv21k_vision_3b/%j.out
#SBATCH --error=logs/training/arxiv21k_vision_3b/%j.err

# Qwen2.5-VL-3B vision on arxiv-21k (main result, 3B variant).

cd "${SLURM_SUBMIT_DIR:-.}"
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1

SHORT_NAME="arxiv21k_vision_3b"
MODALITY="vision"
TRAIN_DATASET="arxiv_50_50_21k_vision_wmetadata_filtered24480_train"
CONFIG_FILE="configs/final_sweep_v7_vision_3b.yaml"
TEMPLATE="qwen2_vl"

PER_DEVICE_BATCH=4
GRAD_ACCUM=2
NUM_GPUS=2
MASTER_PORT=29773

IMG_ARGS="--image_min_pixels 784 --image_max_pixels 1003520"

MODEL_DIR="saves/arxiv_train/small/${SHORT_NAME}"
RESULTS_DIR="results/arxiv_train/small/${SHORT_NAME}"

source sbatch/_common/canonical.sh
