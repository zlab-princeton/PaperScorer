#!/bin/bash
#SBATCH --job-name=paperlens_scaling_text_32b_lora_r128
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gres=gpu:4
#SBATCH --partition=ailab
#SBATCH --output=logs/training/scaling_text_32b_lora_r128/%j.out
#SBATCH --error=logs/training/scaling_text_32b_lora_r128/%j.err

# Model-scaling sweep: Qwen2.5-32B text + LoRA (r=128) on ICLR-balanced.

cd "${SLURM_SUBMIT_DIR:-.}"
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1

SHORT_NAME="scaling_text_32b_lora_r128"
MODALITY="text"
TRAIN_DATASET="iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_train"
CONFIG_FILE="configs/final_sweep_v7_clean_32b_lora_r128.yaml"
TEMPLATE="qwen"

PER_DEVICE_BATCH=2
GRAD_ACCUM=4
NUM_GPUS=4
MASTER_PORT=29710

MODEL_DIR="saves/scaling/${SHORT_NAME}"
RESULTS_DIR="results/scaling/${SHORT_NAME}"

source sbatch/_common/canonical.sh
