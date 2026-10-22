#!/usr/bin/env bash
# Stage B launcher: paperlens-serve ONLY (whole GPU @0.85, no mineru), score the
# prep cache with one or more models (sequential, relaunching per model). Rows
# are pre-cached, so this never touches paperprep.
#
#   MODEL_LINES="1,2,3" srun --partition=gpu-test --gres=gpu:a100:1 \
#     --constraint=gpu80 --qos=gpu-test --time=1:00:00 --mem=64G --cpus-per-task=4 \
#     --job-name=score_a bash scripts/run_score_model.sh
# MODEL_LINES = comma-separated 1-based line numbers into grid_models.tsv.

set -uo pipefail
MODEL_LINES="${MODEL_LINES:?set MODEL_LINES=comma,sep,linenums}"

ROOT=/scratch/gpfs/ZHUANGL/sk7524
WT_VENV=$ROOT/paperlens-training-and-inference/.venv
LF_VENV=$ROOT/LLaMA-Factory-AutoReviewer/.venv
REVIEW=$ROOT/tools/paperlens-reviewing
BASE_CFG=$ROOT/paperlens-training-and-inference/configs/serve.yaml
MODELS_TSV=$REVIEW/scripts/grid_models.tsv
SAVES=$ROOT/LLaMA-Factory-AutoReviewer/saves
CACHE=$REVIEW/logs/grid/prepped
SCORES=$REVIEW/logs/grid/scores
GRID=$REVIEW/logs/grid
mkdir -p "$SCORES"

TAG=$(echo "$MODEL_LINES" | tr ',' '_')
exec > >(tee -a "$GRID/score_${TAG}.log") 2>&1
PORT=18028
echo "[`date +%T`] score job lines=$MODEL_LINES node=$(hostname)"
nvidia-smi -L | head -1

for ln in ${MODEL_LINES//,/ }; do
  IFS=$'\t' read -r LABEL CKREL TEMPLATE MODALITY < <(sed -n "${ln}p" "$MODELS_TSV")
  CKPT="$SAVES/$CKREL"
  if [[ ! -d "$CKPT" ]]; then echo "  [skip line $ln] missing ckpt $CKPT"; continue; fi
  echo ""
  echo "[`date +%T`] === model $LABEL ($MODALITY, $TEMPLATE) ==="
  # Per-model serve config: copy base, swap ckpt_path + template + gpu_mem.
  CFG="$GRID/serve_${LABEL}.yaml"
  "$LF_VENV/bin/python" - "$BASE_CFG" "$CFG" "$CKPT" "$TEMPLATE" <<'PY'
import sys
from omegaconf import OmegaConf
base, out, ckpt, template = sys.argv[1:5]
c = OmegaConf.load(base)
c.model.ckpt_path = ckpt
c.model.template = template
c.vllm.gpu_memory_utilization = 0.85
OmegaConf.save(c, out)
print("wrote", out)
PY

  "$WT_VENV/bin/paperlens" serve --config "$CFG" --port "$PORT" \
      > "$GRID/paperlens_${LABEL}.log" 2>&1 &
  PL_PID=$!
  # wait health
  ok=0
  for _ in {1..180}; do
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then ok=1; break; fi
    if ! kill -0 $PL_PID 2>/dev/null; then echo "  paperlens died during load"; break; fi
    sleep 5
  done
  if [[ $ok -eq 1 ]]; then
    "$LF_VENV/bin/python" "$REVIEW/scripts/score_from_cache.py" \
        --cache-dir "$CACHE" --modality "$MODALITY" \
        --paperlens-url "http://127.0.0.1:$PORT" \
        --model-label "$LABEL" --out "$SCORES/$LABEL.jsonl"
  else
    echo "  [FAIL] $LABEL never became healthy; see $GRID/paperlens_${LABEL}.log"
  fi
  kill $PL_PID 2>/dev/null; sleep 3
  pkill -f "paperlens_cli serve" 2>/dev/null; sleep 2
done
echo "[`date +%T`] score job $MODEL_LINES DONE"
