#!/usr/bin/env bash
# Local dev launcher: starts arxiv_retriever (conda env, port 8001) +
# paperlens-arxiv-server (uv env, port 8000). Logs go to ./logs/.
#
# Usage:
#   bash scripts/launch_local.sh                # start both, foreground
#   PAPERLENS_BG=1 bash scripts/launch_local.sh # both in background
#
# Stop:
#   pkill -f paperlens-arxiv-server
#   pkill -f arxiv_retriever
set -euo pipefail

cd "$(dirname "$(realpath "$0")")/.."
mkdir -p logs

# --- 1) arxiv_retriever (conda env "retriever") ---
RETRIEVER_PORT="${RETRIEVER_PORT:-8001}"
# Config resolution (absolute paths, so retrieval_launch.sh uses them verbatim):
#   explicit RETRIEVER_CFG > the per_venue config `paperlens setup --with-retrieval`
#   generates > the committed per_venue config. (NOT the full-corpus qwen3_06b.yaml.)
ARXIV_SERVER_ROOT="$(pwd)"
if [ -z "${RETRIEVER_CFG:-}" ]; then
    if [ -f "${ARXIV_SERVER_ROOT}/configs/retriever.generated.yaml" ]; then
        RETRIEVER_CFG="${ARXIV_SERVER_ROOT}/configs/retriever.generated.yaml"
    else
        RETRIEVER_CFG="${ARXIV_SERVER_ROOT}/external/arxiv_retriever/configs/retrieval/qwen3_06b_per_venue.yaml"
    fi
fi

echo "[launch] starting arxiv_retriever on :${RETRIEVER_PORT} (conda env: retriever)"
echo "[launch] retriever config: ${RETRIEVER_CFG}"
# Adjust the conda activation command to your system; this assumes miniconda
# is on PATH and an env named `retriever` exists per arxiv_retriever's README.
(
    eval "$(conda shell.bash hook)" 2>/dev/null || true
    conda activate retriever || { echo "ERROR: conda env 'retriever' missing"; exit 2; }
    cd external/arxiv_retriever
    # retrieval_launch.sh takes a POSITIONAL config path + omegaconf overrides
    # (e.g. server.port=8001) -- not --config/--port flags.
    bash src/arxiv_retriever/server/retrieval_launch.sh \
        "${RETRIEVER_CFG}" \
        "server.port=${RETRIEVER_PORT}" 2>&1
) > logs/arxiv_retriever.log 2>&1 &
RETRIEVER_PID=$!
echo "[launch] arxiv_retriever PID=${RETRIEVER_PID}"

# Wait for retriever readiness (max 90s). The server exposes /state (no /health);
# it only answers once the FAISS index + embed model finish loading at startup.
echo -n "[launch] waiting for retriever ..."
for i in {1..45}; do
    if curl -sf "http://localhost:${RETRIEVER_PORT}/state" >/dev/null 2>&1; then
        echo " up"; break
    fi
    sleep 2; echo -n "."
done

# --- 2) paperlens-arxiv-server (uv env) ---
PAPERLENS_PORT="${PAPERLENS_PORT:-8000}"

echo "[launch] starting paperlens-arxiv-server on :${PAPERLENS_PORT}"
if [ -n "${PAPERLENS_BG:-}" ]; then
    (.venv/bin/python -m paperlens_arxiv_server.server --port "${PAPERLENS_PORT}") \
        > logs/paperlens.log 2>&1 &
    PAPERLENS_PID=$!
    echo "[launch] paperlens-arxiv-server PID=${PAPERLENS_PID}"
    echo "[launch] both services running in background. tail -f logs/{arxiv_retriever,paperlens}.log"
else
    .venv/bin/python -m paperlens_arxiv_server.server --port "${PAPERLENS_PORT}"
fi
