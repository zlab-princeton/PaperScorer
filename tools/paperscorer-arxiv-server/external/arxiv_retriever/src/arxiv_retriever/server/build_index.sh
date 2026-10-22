#!/bin/bash
# Usage: ./build_index.sh <model> <corpus_path> <output_dir>
# Example: ./build_index.sh qwen3 /path/to/arxiv.jsonl /path/to/arxiv_index

set -e

# Environment setup
export HF_HOME=/scratch/gpfs/ZHUANGL/sk7524/hf
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Local model paths
E5_MODEL_PATH=/scratch/gpfs/ZHUANGL/sk7524/hf/hub/models--intfloat--e5-base-v2/snapshots/f52bf8ec8c7124536f0efb74aca902b2995e5bcd
QWEN3_MODEL_PATH=/scratch/gpfs/ZHUANGL/sk7524/hf/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/c54f2e6e80b2d7b7de06f51cec4959f6b3e03418

if [ $# -lt 3 ]; then
    echo "Usage: $0 <model> <corpus_path> <output_dir>"
    echo "  model:       e5 or qwen3"
    echo "  corpus_path: path to .jsonl corpus file"
    echo "  output_dir:  directory to save index"
    echo ""
    echo "Example: $0 qwen3 /path/to/wiki-18.jsonl /path/to/wiki_qwen3_index"
    exit 1
fi

MODEL=$1
CORPUS_PATH=$2
OUTPUT_DIR=$3

# Validate corpus exists
if [ ! -f "$CORPUS_PATH" ]; then
    echo "Error: Corpus file not found: $CORPUS_PATH"
    exit 1
fi

# Model-specific configuration
case $MODEL in
    e5)
        retriever_name=e5
        retriever_model=$E5_MODEL_PATH
        pooling_method=mean
        ;;
    qwen3)
        retriever_name=qwen3
        retriever_model=$QWEN3_MODEL_PATH
        pooling_method=last_token
        ;;
    *)
        echo "Error: Unknown model '$MODEL'. Supported: e5, qwen3"
        exit 1
        ;;
esac

echo "Building index..."
echo "  Model:  $retriever_name ($retriever_model)"
echo "  Corpus: $CORPUS_PATH"
echo "  Output: $OUTPUT_DIR"
echo "  Pooling: $pooling_method"

mkdir -p $OUTPUT_DIR

python examples/search/retriever/index_builder.py \
    --retrieval_method $retriever_name \
    --model_path $retriever_model \
    --corpus_path $CORPUS_PATH \
    --save_dir $OUTPUT_DIR \
    --use_fp16 \
    --max_length 256 \
    --batch_size 384 \
    --pooling_method $pooling_method \
    --faiss_type Flat \
    --save_embedding \
    --faiss_gpu

echo "Done! Index saved to: $OUTPUT_DIR/${retriever_name}_Flat.index"
