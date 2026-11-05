#!/bin/bash
#SBATCH --job-name=paperlens_scaling_vision_32b
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --gres=gpu:4
#SBATCH --partition=ailab
#SBATCH --output=logs/training/scaling_vision_32b/%j.out
#SBATCH --error=logs/training/scaling_vision_32b/%j.err

# Model-scaling sweep: Qwen2.5-VL-32B vision (full FT) on ICLR-balanced.

cd "${SLURM_SUBMIT_DIR:-.}"
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1

SHORT_NAME="scaling_vision_32b"
MODALITY="vision"
TRAIN_DATASET="iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_labelfix_v7_filtered_filtered24480_train"
CONFIG_FILE="configs/final_sweep_v7_vision_32b.yaml"
TEMPLATE="qwen2_vl"

PER_DEVICE_BATCH=1
GRAD_ACCUM=4
NUM_GPUS=4
MASTER_PORT=29705

IMG_ARGS="--image_min_pixels 784 --image_max_pixels 1003520"

MODEL_DIR="saves/scaling/${SHORT_NAME}"
RESULTS_DIR="results/scaling/${SHORT_NAME}"

source sbatch/_common/canonical.sh
