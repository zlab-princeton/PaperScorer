#!/bin/bash
# Launch the arxiv_retriever FAISS server under the existing `retriever` conda env.
#
# Usage:
#   retrieval_launch.sh <config_path> [omegaconf_overrides...]
#
# Example:
#   retrieval_launch.sh configs/retrieval/qwen3_4b.yaml
#   retrieval_launch.sh configs/retrieval/qwen3_4b.yaml server.port=8001
#
# Assumes the caller has already:
#   - module load anaconda3/2024.6
#   - conda activate retriever
#   - pip install -e /path/to/arxiv_retriever (so `python -m arxiv_retriever.server.retrieval_server` works)

set -e

CONFIG="${1:?Usage: retrieval_launch.sh <config_path> [overrides...]}"
shift

# Resolve config path: if relative, resolve relative to the repo root (two levels up from this script)
if [ ! -f "$CONFIG" ]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
    if [ -f "$REPO_ROOT/$CONFIG" ]; then
        CONFIG="$REPO_ROOT/$CONFIG"
    else
        echo "Error: config not found: $CONFIG (also tried $REPO_ROOT/$CONFIG)"
        exit 1
    fi
fi

# Log directory
LOG_DIR="${ARXIV_RETRIEVER_LOG_DIR:-logs/retrieval_server}"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/$(date '+%Y%m%d_%H%M%S')_$$.log"

echo "Starting arxiv_retriever server..."
echo "  Config: $CONFIG"
echo "  Overrides: $*"
echo "  Log: $LOG_FILE"

# Launch in background — use direct path so it works without pip install -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_SCRIPT="$SCRIPT_DIR/retrieval_server.py"
nohup python "$SERVER_SCRIPT" --config "$CONFIG" "$@" > "$LOG_FILE" 2>&1 &
SERVER_PID=$!
echo "Server started (PID: $SERVER_PID)"
echo "$SERVER_PID" > "$LOG_DIR/server.pid"
echo ""
echo "Test with:"
echo "  curl -X POST http://localhost:8000/retrieve -H 'Content-Type: application/json' \\"
echo "    -d '{\"query\": \"What is Python?\", \"topk\": 3}'"
echo ""
echo "View logs:"
echo "  tail -f $LOG_FILE"
