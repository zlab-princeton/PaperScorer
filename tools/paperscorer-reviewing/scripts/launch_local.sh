#!/usr/bin/env bash
# launch_local.sh — bring up the full reviewing stack on the local host.
#
#   1. Verify nvidia-smi finds a CUDA GPU.
#   2. Launch `paperprep serve` (PaperLens/paperprep, persistent MinerU vLLM)
#      in background and wait for its /healthz.
#   3. Launch `paperlens serve` (paperlens-training-and-inference) in background
#      and wait for its /health.
#   4. Launch `paperlensreview serve` in foreground; print its URL.
#
# Stop:
#   pkill -f "paperprep .*serve"
#   pkill -f "paperlens serve"
#   pkill -f "paperlensreview"
#
# GPU note: both paperprep-serve (mineru-vllm-server, ~1.2B) and paperlens-serve
# (Qwen2-VL 3B) hold a vLLM engine on the same GPU. Default split is 0.3 for
# paperprep and whatever paperlens-serve's serve.yaml says — confirm the two
# add to <1.0 or you'll OOM. Override with PAPERPREP_GPU_MEM_UTIL.
#
# Env / overrides:
#   PAPERLENS_SERVE_CONFIG          path to paperlens-training-and-inference/configs/serve.yaml
#   PAPERLENSREVIEW_CONFIG          path to paperlens-reviewing/configs/server.yaml
#   PAPERPREP_SERVE_OUTPUT_DIR      where paperprep serve writes per-request subdirs
#   PAPERLENS_SERVE_PORT            default 8002
#   PAPERLENSREVIEW_PORT            default 8003
#   PAPERPREP_SERVE_PORT            default 8004
#   PAPERPREP_MINERU_PORT           default 30000 (internal mineru-vllm-server)
#   PAPERPREP_GPU_MEM_UTIL          default 0.30
#   PAPERLENS_TRAIN_AND_INFER_VENV  venv that has the `paperlens` CLI (or src tree)
#   PAPERPREP_VENV                  venv that has the `paperprep` CLI
#   PAPERLENSREVIEW_VENV            venv that has the `paperlensreview` CLI
#   PAPERPREP_ROOT                  paperprep repo root (auto-discovered if unset)

set -euo pipefail

HERE="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
REVIEW_ROOT="$(cd "$HERE/.." && pwd)"

# ----- 0. Defaults --------------------------------------------------------
PAPERLENS_SERVE_PORT="${PAPERLENS_SERVE_PORT:-8002}"
PAPERLENSREVIEW_PORT="${PAPERLENSREVIEW_PORT:-8003}"
PAPERPREP_SERVE_PORT="${PAPERPREP_SERVE_PORT:-8004}"
PAPERPREP_MINERU_PORT="${PAPERPREP_MINERU_PORT:-30000}"
PAPERPREP_GPU_MEM_UTIL="${PAPERPREP_GPU_MEM_UTIL:-0.30}"
PAPERPREP_SERVE_OUTPUT_DIR="${PAPERPREP_SERVE_OUTPUT_DIR:-$REVIEW_ROOT/paperprep_serve_work}"

_pick_free_port() {
    local preferred="$1"
    python3 - "$preferred" <<'PY'
import socket, sys
preferred = int(sys.argv[1])
def is_free(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("", port))
        except OSError:
            return False
    return True
if is_free(preferred):
    print(preferred)
else:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        print(s.getsockname()[1])
PY
}

REQUESTED_PAPERLENS_SERVE_PORT="$PAPERLENS_SERVE_PORT"
REQUESTED_PAPERLENSREVIEW_PORT="$PAPERLENSREVIEW_PORT"
REQUESTED_PAPERPREP_SERVE_PORT="$PAPERPREP_SERVE_PORT"
PAPERLENS_SERVE_PORT="$(_pick_free_port "$PAPERLENS_SERVE_PORT")"
PAPERLENSREVIEW_PORT="$(_pick_free_port "$PAPERLENSREVIEW_PORT")"
PAPERPREP_SERVE_PORT="$(_pick_free_port "$PAPERPREP_SERVE_PORT")"
[[ "$PAPERLENS_SERVE_PORT"  != "$REQUESTED_PAPERLENS_SERVE_PORT"  ]] && echo "[launcher] paperlens serve: port $REQUESTED_PAPERLENS_SERVE_PORT busy, using $PAPERLENS_SERVE_PORT"
[[ "$PAPERLENSREVIEW_PORT"  != "$REQUESTED_PAPERLENSREVIEW_PORT"  ]] && echo "[launcher] paperlensreview:  port $REQUESTED_PAPERLENSREVIEW_PORT busy, using $PAPERLENSREVIEW_PORT"
[[ "$PAPERPREP_SERVE_PORT"  != "$REQUESTED_PAPERPREP_SERVE_PORT"  ]] && echo "[launcher] paperprep serve: port $REQUESTED_PAPERPREP_SERVE_PORT busy, using $PAPERPREP_SERVE_PORT"

# Reviewing server reads its upstream base_urls from config; override via env
# so dynamic ports propagate without rewriting any yaml.
export PAPERLENS_SERVE_BASE_URL="http://127.0.0.1:${PAPERLENS_SERVE_PORT}"
export PAPERPREP_SERVE_BASE_URL="http://127.0.0.1:${PAPERPREP_SERVE_PORT}"

# Best-effort discovery of paperlens-training-and-inference.
if [[ -z "${PAPERLENS_SERVE_CONFIG:-}" ]]; then
    for candidate in \
        "$REVIEW_ROOT/../../paperlens-training-and-inference/configs/serve.yaml" \
        "$REVIEW_ROOT/../paperlens-training-and-inference/configs/serve.yaml" \
        "/scratch/gpfs/ZHUANGL/sk7524/paperlens-training-and-inference/configs/serve.yaml"
    do
        if [[ -f "$candidate" ]]; then PAPERLENS_SERVE_CONFIG="$candidate"; break; fi
    done
fi
if [[ -z "${PAPERLENS_SERVE_CONFIG:-}" || ! -f "$PAPERLENS_SERVE_CONFIG" ]]; then
    echo "ERROR: PAPERLENS_SERVE_CONFIG not set and no sibling config found." >&2
    exit 2
fi
PAPERLENSREVIEW_CONFIG="${PAPERLENSREVIEW_CONFIG:-$REVIEW_ROOT/configs/server.yaml}"

# Best-effort discovery of paperprep root.
if [[ -z "${PAPERPREP_ROOT:-}" ]]; then
    for candidate in \
        "/scratch/gpfs/ZHUANGL/sk7524/PaperLens/paperprep" \
        "$REVIEW_ROOT/../../PaperLens/paperprep" \
        "$REVIEW_ROOT/../paperprep"
    do
        if [[ -d "$candidate/src/paper_anonymizer/paperprep" ]]; then
            PAPERPREP_ROOT="$(cd "$candidate" && pwd)"; break
        fi
    done
fi
if [[ -z "${PAPERPREP_ROOT:-}" || ! -d "$PAPERPREP_ROOT/src/paper_anonymizer/paperprep" ]]; then
    echo "ERROR: PAPERPREP_ROOT not set and paperprep submodule not found." >&2
    echo "Clone https://github.com/SachinKonan/AutoReviewer (paperprep-only branch)" >&2
    echo "or set PAPERPREP_ROOT=<repo root>." >&2
    exit 2
fi

mkdir -p "$PAPERPREP_SERVE_OUTPUT_DIR"

echo "[launcher] review root:           $REVIEW_ROOT"
echo "[launcher] paperprep root:        $PAPERPREP_ROOT  (port $PAPERPREP_SERVE_PORT, mineru :$PAPERPREP_MINERU_PORT)"
echo "[launcher] paperlens serve cfg:   $PAPERLENS_SERVE_CONFIG  (port $PAPERLENS_SERVE_PORT)"
echo "[launcher] paperlensreview cfg:   $PAPERLENSREVIEW_CONFIG  (port $PAPERLENSREVIEW_PORT)"

LOG_DIR="${LOG_DIR:-$REVIEW_ROOT/logs}"
mkdir -p "$LOG_DIR"

# ----- 1. NVIDIA GPU sanity ----------------------------------------------
echo ""
echo "[launcher] step 1/4: NVIDIA GPU check"
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "  ERROR: nvidia-smi not on PATH; paperprep + paperlens serve both need a CUDA GPU." >&2
    exit 2
fi
GPU_LINE=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)
[[ -z "$GPU_LINE" ]] && { echo "  ERROR: nvidia-smi found no GPUs." >&2; exit 2; }
echo "  ✓ GPU: $GPU_LINE"

# ----- 2. Launch paperprep serve -----------------------------------------
echo ""
echo "[launcher] step 2/4: launch paperprep serve  (gpu-memory-utilization=$PAPERPREP_GPU_MEM_UTIL)"
PAPERPREP_LOG="$LOG_DIR/paperprep_serve.log"

PAPERPREP_VENV_OPT="${PAPERPREP_VENV:-$PAPERPREP_ROOT/.venv}"
PAPERPREP_PY=""
if [[ -x "$PAPERPREP_VENV_OPT/bin/python" ]]; then
    PAPERPREP_PY="$PAPERPREP_VENV_OPT/bin/python"
    echo "  using paperprep venv: $PAPERPREP_VENV_OPT"
elif command -v paperprep >/dev/null 2>&1; then
    PAPERPREP_PY=""  # use console script directly
    echo "  using 'paperprep' on PATH: $(command -v paperprep)"
else
    echo "  ERROR: cannot locate paperprep. Install with 'cd $PAPERPREP_ROOT && uv sync --extra gpu --extra serve'" >&2
    exit 2
fi

if curl -sf "http://127.0.0.1:${PAPERPREP_SERVE_PORT}/healthz" >/dev/null 2>&1; then
    echo "  ✓ paperprep serve already up on :${PAPERPREP_SERVE_PORT}"
else
    # Scope the PATH override to just the paperprep subprocess: paperprep serve
    # shells out to `mineru-vllm-server` (a binary in the paperprep venv's bin/),
    # but the launcher must keep its own PATH clean so step 3's python lookup
    # still resolves to the parent shell's (LF) venv, which has peft + vllm
    # + transformers needed by paperlens-serve.
    if [[ -n "$PAPERPREP_PY" ]]; then
        PAPERPREP_BIN_DIR="$(dirname "$PAPERPREP_PY")"
        PAPERPREP_LAUNCH_CMD=(env "PATH=$PAPERPREP_BIN_DIR:$PATH" "$PAPERPREP_PY" -m paper_anonymizer.paperprep.cli serve)
    else
        PAPERPREP_LAUNCH_CMD=(paperprep serve)
    fi
    echo "  starting paperprep serve, logs -> $PAPERPREP_LOG"
    nohup "${PAPERPREP_LAUNCH_CMD[@]}" \
        --output-dir "$PAPERPREP_SERVE_OUTPUT_DIR" \
        --host 127.0.0.1 \
        --port "$PAPERPREP_SERVE_PORT" \
        --mineru-port "$PAPERPREP_MINERU_PORT" \
        --gpu-memory-utilization "$PAPERPREP_GPU_MEM_UTIL" \
        >"$PAPERPREP_LOG" 2>&1 &
    PAPERPREP_PID=$!
    echo "  paperprep serve PID: $PAPERPREP_PID"
    echo "  waiting for /healthz (up to 10 min — vLLM warmup) ..."
    for _ in {1..120}; do
        if curl -sf "http://127.0.0.1:${PAPERPREP_SERVE_PORT}/healthz" >/dev/null 2>&1; then
            echo "  ✓ paperprep serve healthy"; break
        fi
        sleep 5
    done
    if ! curl -sf "http://127.0.0.1:${PAPERPREP_SERVE_PORT}/healthz" >/dev/null 2>&1; then
        echo "  ERROR: paperprep serve never came up. Check $PAPERPREP_LOG" >&2
        exit 1
    fi
fi

# ----- 3. Launch paperlens serve -----------------------------------------
echo ""
echo "[launcher] step 3/4: launch paperlens serve"
PAPERLENS_LOG="$LOG_DIR/paperlens_serve.log"

PAPERLENS_VENV="${PAPERLENS_TRAIN_AND_INFER_VENV:-}"
# Auto-discover the canonical LF venv if the caller didn't set it. paperlens-serve
# imports peft + vllm + transformers; the system python3.9 doesn't have these.
if [[ -z "$PAPERLENS_VENV" ]]; then
    for candidate in \
        "/scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer/.venv" \
        "${VIRTUAL_ENV:-}"
    do
        if [[ -n "$candidate" && -x "$candidate/bin/python" ]]; then
            PAPERLENS_VENV="$candidate"; break
        fi
    done
fi
if [[ -n "$PAPERLENS_VENV" && -f "$PAPERLENS_VENV/bin/activate" ]]; then
    # shellcheck disable=SC1090,SC1091
    source "$PAPERLENS_VENV/bin/activate"
    echo "  activated paperlens venv: $PAPERLENS_VENV"
fi

PAPERLENS_TRAIN_AND_INFER_ROOT="${PAPERLENS_TRAIN_AND_INFER_ROOT:-}"
if [[ -z "$PAPERLENS_TRAIN_AND_INFER_ROOT" ]]; then
    for candidate in \
        "$(dirname "$PAPERLENS_SERVE_CONFIG")/.." \
        "$REVIEW_ROOT/../../paperlens-training-and-inference" \
        "$REVIEW_ROOT/../paperlens-training-and-inference" \
        "/scratch/gpfs/ZHUANGL/sk7524/paperlens-training-and-inference"
    do
        if [[ -d "$candidate/src/paperlens_cli" ]]; then
            PAPERLENS_TRAIN_AND_INFER_ROOT="$(cd "$candidate" && pwd)"; break
        fi
    done
fi

PAPERLENS_LAUNCH_CMD=()
if command -v paperlens >/dev/null 2>&1; then
    PAPERLENS_LAUNCH_CMD=(paperlens serve)
    echo "  using 'paperlens' CLI at $(command -v paperlens)"
elif [[ -n "$PAPERLENS_TRAIN_AND_INFER_ROOT" && -d "$PAPERLENS_TRAIN_AND_INFER_ROOT/src/paperlens_cli" ]]; then
    PYBIN="$(command -v python3 || command -v python || true)"
    [[ -z "$PYBIN" ]] && { echo "  ERROR: no python on PATH for paperlens fallback." >&2; exit 2; }
    PAPERLENS_LAUNCH_CMD=(env "PYTHONPATH=$PAPERLENS_TRAIN_AND_INFER_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$PYBIN" -m paperlens_cli serve)
    echo "  using fallback: python -m paperlens_cli  (root=$PAPERLENS_TRAIN_AND_INFER_ROOT)"
else
    echo "  ERROR: cannot locate the paperlens CLI." >&2; exit 2
fi

if curl -sf "http://127.0.0.1:${PAPERLENS_SERVE_PORT}/health" >/dev/null 2>&1; then
    echo "  ✓ paperlens serve already up on :${PAPERLENS_SERVE_PORT}"
else
    echo "  starting paperlens serve, logs -> $PAPERLENS_LOG"
    nohup "${PAPERLENS_LAUNCH_CMD[@]}" \
        --config "$PAPERLENS_SERVE_CONFIG" \
        --port "$PAPERLENS_SERVE_PORT" \
        >"$PAPERLENS_LOG" 2>&1 &
    PAPERLENS_PID=$!
    echo "  paperlens serve PID: $PAPERLENS_PID"
    echo "  waiting for /health (up to 10 min) ..."
    for _ in {1..120}; do
        if curl -sf "http://127.0.0.1:${PAPERLENS_SERVE_PORT}/health" >/dev/null 2>&1; then
            echo "  ✓ paperlens serve healthy"; break
        fi
        sleep 5
    done
    if ! curl -sf "http://127.0.0.1:${PAPERLENS_SERVE_PORT}/health" >/dev/null 2>&1; then
        echo "  ERROR: paperlens serve never came up. Check $PAPERLENS_LOG" >&2
        exit 1
    fi
fi

# NOTE: deliberately do NOT deactivate the paperlens venv here — paperlensreview
# imports fastapi/uvicorn/omegaconf/etc. from the same LF venv. If the caller
# wants a different venv for the reviewing server, set PAPERLENSREVIEW_VENV.

# ----- 4. Launch paperlensreview server ----------------------------------
echo ""
echo "[launcher] step 4/4: launch paperlensreview"
REVIEW_VENV="${PAPERLENSREVIEW_VENV:-}"
# Auto-discover: prefer the explicit env, then $VIRTUAL_ENV (set by step 3),
# then the canonical LF venv.
if [[ -z "$REVIEW_VENV" ]]; then
    for candidate in "${VIRTUAL_ENV:-}" "/scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer/.venv"; do
        if [[ -n "$candidate" && -x "$candidate/bin/python" ]]; then
            REVIEW_VENV="$candidate"; break
        fi
    done
fi
if [[ -n "$REVIEW_VENV" && -f "$REVIEW_VENV/bin/activate" && "$REVIEW_VENV" != "${VIRTUAL_ENV:-}" ]]; then
    # shellcheck disable=SC1090,SC1091
    source "$REVIEW_VENV/bin/activate"
fi
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    echo "  paperlensreview venv: $VIRTUAL_ENV"
fi

REVIEW_LAUNCH_CMD=()
if command -v paperlensreview >/dev/null 2>&1; then
    REVIEW_LAUNCH_CMD=(paperlensreview serve)
    echo "  using 'paperlensreview' CLI at $(command -v paperlensreview)"
elif [[ -d "$REVIEW_ROOT/src/paperlensreview" ]]; then
    PYBIN="$(command -v python3 || command -v python || true)"
    [[ -z "$PYBIN" ]] && { echo "  ERROR: no python on PATH for paperlensreview fallback." >&2; exit 2; }
    REVIEW_LAUNCH_CMD=(env "PYTHONPATH=$REVIEW_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$PYBIN" -m paperlensreview serve)
    echo "  using fallback: python -m paperlensreview  (root=$REVIEW_ROOT)"
else
    echo "  ERROR: paperlensreview CLI not on PATH and src tree missing" >&2
    exit 2
fi

URL="http://$(hostname):${PAPERLENSREVIEW_PORT}"
echo ""
echo "================================================================"
echo "  🌐 PaperLens reviewing UI ready at:"
echo "     $URL"
echo "================================================================"
echo ""

exec "${REVIEW_LAUNCH_CMD[@]}" \
    --config "$PAPERLENSREVIEW_CONFIG" \
    --port "$PAPERLENSREVIEW_PORT"
