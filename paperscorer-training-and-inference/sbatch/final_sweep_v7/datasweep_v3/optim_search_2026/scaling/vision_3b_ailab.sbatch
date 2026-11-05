#!/bin/bash
#SBATCH --job-name=paperlens_scaling_vision_3b
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=350G
#SBATCH --gres=gpu:2
#SBATCH --partition=ailab
#SBATCH --output=logs/training/scaling_vision_3b/%j.out
#SBATCH --error=logs/training/scaling_vision_3b/%j.err

# Model-scaling sweep: Qwen2.5-VL-3B vision on ICLR-balanced.

cd "${SLURM_SUBMIT_DIR:-.}"
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1

SHORT_NAME="scaling_vision_3b"
MODALITY="vision"
TRAIN_DATASET="iclr_2020_2023_2025_2026_85_5_10_balanced_original_vision_labelfix_v7_filtered_filtered24480_train"
CONFIG_FILE="configs/final_sweep_v7_vision_3b.yaml"
TEMPLATE="qwen2_vl"

PER_DEVICE_BATCH=4
GRAD_ACCUM=2
NUM_GPUS=2
MASTER_PORT=29704

IMG_ARGS="--image_min_pixels 784 --image_max_pixels 1003520"

MODEL_DIR="saves/scaling/${SHORT_NAME}"
RESULTS_DIR="results/scaling/${SHORT_NAME}"

source sbatch/_common/canonical.sh
