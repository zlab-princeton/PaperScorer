#!/bin/bash
#SBATCH --job-name=paperlens_arxiv21k_vision_7b
#SBATCH --time=18:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --gres=gpu:4
#SBATCH --partition=ailab
#SBATCH --output=logs/training/arxiv21k_vision_7b/%j.out
#SBATCH --error=logs/training/arxiv21k_vision_7b/%j.err

# Qwen2.5-VL-7B vision on arxiv-21k (main result).

cd "${SLURM_SUBMIT_DIR:-.}"
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1

SHORT_NAME="arxiv21k_vision_7b"
MODALITY="vision"
TRAIN_DATASET="arxiv_50_50_21k_vision_wmetadata_filtered24480_train"
CONFIG_FILE="configs/final_sweep_v7_vision.yaml"
TEMPLATE="qwen2_vl"

PER_DEVICE_BATCH=4
GRAD_ACCUM=1
NUM_GPUS=4
MASTER_PORT=29772

IMG_ARGS="--image_min_pixels 784 --image_max_pixels 1003520"

MODEL_DIR="saves/arxiv_train/small/${SHORT_NAME}"
RESULTS_DIR="results/arxiv_train/small/${SHORT_NAME}"

source sbatch/_common/canonical.sh
