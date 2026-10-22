#!/usr/bin/env bash
# Self-contained GPU job: bring up paperprep-serve + paperlens-serve, score ONE
# SLICE of the PaperLens paper's zlab-era commits (f48b7f8..HEAD), tear down.
# Two of these run in parallel on separate gpu-test GPUs (15 commits each, <1h).
# A merge+plot step runs afterward once both slices land.
#
#   SLICE_INDEX=0 NUM_SLICES=2 srun --partition=gpu-test --gres=gpu:a100:1 \
#       --constraint=gpu80 --qos=gpu-test --time=1:00:00 --mem=64G --cpus-per-task=4 \
#       --job-name=ctrack0 bash scripts/run_commit_track.sh
#
# Env: SLICE_INDEX (default 0), NUM_SLICES (default 1).

set -uo pipefail

SLICE_INDEX="${SLICE_INDEX:-0}"
NUM_SLICES="${NUM_SLICES:-1}"

ROOT=/scratch/gpfs/ZHUANGL/sk7524
LF_VENV=$ROOT/LLaMA-Factory-AutoReviewer/.venv
WT_VENV=$ROOT/paperlens-training-and-inference/.venv
PP_ROOT=$ROOT/PaperLens/paperprep
REVIEW=$ROOT/tools/paperlens-reviewing
PAPER_REPO=$ROOT/PaperLensArXivRelease
SERVE_CFG=$ROOT/paperlens-training-and-inference/configs/serve.yaml

OUT_DIR=$REVIEW/logs/commit_track
mkdir -p "$OUT_DIR"
exec > >(tee -a "$OUT_DIR/run_slice${SLICE_INDEX}.log") 2>&1

# Per-slice ports so two jobs co-located on one node wouldn't collide
# (GPU would still be shared then — gpu-test normally gives separate nodes).
PAPERPREP_PORT=$((18024 + SLICE_INDEX))
PAPERLENS_PORT=$((18022 + SLICE_INDEX))
MINERU_PORT=$((40010 + SLICE_INDEX * 5))
RESULTS_NAME="results_slice${SLICE_INDEX}.jsonl"

echo "[`date +%T`] slice ${SLICE_INDEX}/${NUM_SLICES} node=$(hostname) ports pp=$PAPERPREP_PORT pl=$PAPERLENS_PORT"
nvidia-smi -L | head -1

# ---- paperprep serve (latex_dir compile path uses our pdflatex+bibtex fallback) ----
echo "[`date +%T`] launch paperprep serve (gpu-mem 0.20)"
env "PATH=$PP_ROOT/.venv/bin:$PATH" "$PP_ROOT/.venv/bin/python" \
    -m paper_anonymizer.paperprep.cli serve \
    --output-dir "$OUT_DIR/paperprep_work_slice${SLICE_INDEX}" \
    --host 127.0.0.1 --port "$PAPERPREP_PORT" \
    --mineru-port "$MINERU_PORT" \
    --gpu-memory-utilization 0.20 \
    > "$OUT_DIR/paperprep_slice${SLICE_INDEX}.log" 2>&1 &
PP_PID=$!

# ---- paperlens serve (worktree venv, ckpt per serve.yaml = 7B) ----
echo "[`date +%T`] launch paperlens serve"
"$WT_VENV/bin/paperlens" serve --config "$SERVE_CFG" --port "$PAPERLENS_PORT" \
    > "$OUT_DIR/paperlens_slice${SLICE_INDEX}.log" 2>&1 &
PL_PID=$!

trap 'echo "[cleanup] killing daemons"; kill $PL_PID $PP_PID 2>/dev/null; pkill -f mineru-vllm-server 2>/dev/null; exit' EXIT

echo "[`date +%T`] waiting for both /health ..."
for _ in {1..360}; do
  if curl -sf "http://127.0.0.1:$PAPERPREP_PORT/healthz" >/dev/null 2>&1 \
     && curl -sf "http://127.0.0.1:$PAPERLENS_PORT/health" >/dev/null 2>&1; then
    echo "[`date +%T`] both healthy"; break
  fi
  sleep 5
done
curl -sf "http://127.0.0.1:$PAPERLENS_PORT/health" >/dev/null 2>&1 || { echo "paperlens never came up"; exit 1; }

# ---- score this slice of the zlab-era commits ----
echo "[`date +%T`] scoring slice ${SLICE_INDEX}/${NUM_SLICES} of f48b7f8..HEAD"
"$LF_VENV/bin/python" "$REVIEW/scripts/score_commits.py" \
    --repo "$PAPER_REPO" \
    --from-commit f48b7f8 \
    --num-slices "$NUM_SLICES" --slice-index "$SLICE_INDEX" \
    --results-name "$RESULTS_NAME" \
    --modality vision \
    --paperprep-url "http://127.0.0.1:$PAPERPREP_PORT" \
    --paperlens-url "http://127.0.0.1:$PAPERLENS_PORT" \
    --out-dir "$OUT_DIR"

echo "[`date +%T`] slice ${SLICE_INDEX} DONE -> $OUT_DIR/$RESULTS_NAME"
