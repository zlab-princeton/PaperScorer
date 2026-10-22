#!/usr/bin/env bash
# Stage A launcher: paperprep-serve ONLY (no paperlens co-resident -> stable,
# gpu_mem 0.85), prep one slice of all grid papers into the cache.
#
#   SLICE_INDEX=0 NUM_SLICES=3 srun --partition=gpu-test --gres=gpu:a100:1 \
#     --constraint=gpu80 --qos=gpu-test --time=1:00:00 --mem=64G --cpus-per-task=4 \
#     --job-name=prep0 bash scripts/run_prep_cache.sh

set -uo pipefail
SLICE_INDEX="${SLICE_INDEX:-0}"
NUM_SLICES="${NUM_SLICES:-1}"

ROOT=/scratch/gpfs/ZHUANGL/sk7524
LF_VENV=$ROOT/LLaMA-Factory-AutoReviewer/.venv
PP_ROOT=$ROOT/PaperLens/paperprep
REVIEW=$ROOT/tools/paperlens-reviewing
PAPER_REPO=$ROOT/PaperLensArXivRelease
TEST_PDFS=$ROOT/LLaMA-Factory-AutoReviewer/test_pdfs
NEURIPS_PDF=$ROOT/PaperLens---NIPS26---COLM-port/neurips_2026.pdf
COLM_PDF=$ROOT/AutoReviewer-COLM2026/colm2026_conference.pdf
CACHE=$REVIEW/logs/grid/prepped

GRID=$REVIEW/logs/grid
mkdir -p "$GRID"
exec > >(tee -a "$GRID/prep_slice${SLICE_INDEX}.log") 2>&1

PORT=$((18040 + SLICE_INDEX))
MINERU_PORT=$((40040 + SLICE_INDEX * 5))
echo "[`date +%T`] prep slice ${SLICE_INDEX}/${NUM_SLICES} node=$(hostname) port=$PORT"
nvidia-smi -L | head -1

env "PATH=$PP_ROOT/.venv/bin:$PATH" "$PP_ROOT/.venv/bin/python" \
    -m paper_anonymizer.paperprep.cli serve \
    --output-dir "$GRID/paperprep_work_slice${SLICE_INDEX}" \
    --host 127.0.0.1 --port "$PORT" --mineru-port "$MINERU_PORT" \
    --gpu-memory-utilization 0.85 > "$GRID/paperprep_slice${SLICE_INDEX}.log" 2>&1 &
PP_PID=$!
trap 'kill $PP_PID 2>/dev/null; pkill -f mineru-vllm-server 2>/dev/null; exit' EXIT

echo "[`date +%T`] waiting for /healthz ..."
for _ in {1..360}; do
  curl -sf "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1 && break; sleep 5
done
curl -sf "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1 || { echo "paperprep never came up"; exit 1; }

"$LF_VENV/bin/python" "$REVIEW/scripts/build_prep_cache.py" \
    --cache-dir "$CACHE" \
    --repo "$PAPER_REPO" --from-commit f48b7f8 \
    --test-pdfs-root "$TEST_PDFS" \
    --extra-pdf "neurips=$NEURIPS_PDF" \
    --extra-pdf "colm=$COLM_PDF" \
    --paperprep-url "http://127.0.0.1:$PORT" \
    --num-slices "$NUM_SLICES" --slice-index "$SLICE_INDEX"

echo "[`date +%T`] prep slice ${SLICE_INDEX} DONE"
