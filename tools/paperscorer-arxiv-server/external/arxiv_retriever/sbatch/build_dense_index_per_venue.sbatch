#!/bin/bash
# Build a Qwen3-Embedding-0.6B faiss (Flat) index over the per_venue subset
# (~80K papers) for LitSearch dense-retrieval comparisons.
#
# Submit:
#   sbatch arxiv_retriever/sbatch/build_dense_index_per_venue.sbatch
#
# Output: $OUT_DIR/  (qwen3_Flat.index + emb_qwen3.memmap)
#
#SBATCH --job-name=dense_pv_index
#SBATCH --partition=gpu-test
#SBATCH --constraint=gpu80
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:55:00
#SBATCH --output=logs/dense_pv_index_%j.out
#SBATCH --error=logs/dense_pv_index_%j.err

set -uo pipefail

WORKDIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$WORKDIR"
mkdir -p logs

SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/gpfs/ZHUANGL/sk7524}"
export HF_HOME="${HF_HOME:-${SCRATCH_ROOT}/hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${SCRATCH_ROOT}/hf}"
export TRANSFORMERS_OFFLINE=1

SUBSET_JSONL="${SUBSET_JSONL:-${SCRATCH_ROOT}/arxiv_bm25_per_venue_stage/arxiv_wikiformat_per_venue.jsonl}"
OUT_DIR="${OUT_DIR:-${SCRATCH_ROOT}/arxiv_faiss_per_venue_index}"
MODEL_PATH="${MODEL_PATH:-${SCRATCH_ROOT}/hf/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/c54f2e6e80b2d7b7de06f51cec4959f6b3e03418}"
RETRIEVER_CONDA_ENV="${RETRIEVER_CONDA_ENV:-retriever}"

echo "=============================================="
echo "Job:        $SLURM_JOB_ID"
echo "Host:       $(hostname)"
echo "Subset:     $SUBSET_JSONL  ($(wc -l < "$SUBSET_JSONL") docs)"
echo "Model:      $MODEL_PATH"
echo "Output:     $OUT_DIR"
echo "=============================================="
nvidia-smi

test -f "$SUBSET_JSONL" || { echo "ERROR: subset jsonl missing: $SUBSET_JSONL"; exit 1; }
test -d "$MODEL_PATH"   || { echo "ERROR: model path missing: $MODEL_PATH"; exit 1; }

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

set +u
module load anaconda3/2024.6
conda activate "$RETRIEVER_CONDA_ENV"
set -u

echo ""
echo "=== building Qwen3-0.6B faiss index (per_venue subset) ==="
python src/arxiv_retriever/server/index_builder.py \
    --retrieval_method qwen3 \
    --model_path "$MODEL_PATH" \
    --corpus_path "$SUBSET_JSONL" \
    --save_dir "$OUT_DIR" \
    --max_length 256 \
    --batch_size 512 \
    --use_fp16 \
    --pooling_method last_token \
    --save_embedding \
    --faiss_type Flat \
    --faiss_gpu

echo ""
echo "=== done ==="
du -sh "$OUT_DIR"
ls -lh "$OUT_DIR"
