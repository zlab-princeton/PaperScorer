#!/usr/bin/env bash
# 27-PDF batch under the 7B paperlens-serve (vs the 3B-tested-yesterday).
# Mirrors test_worktree_venv.sh except for the 7B ckpt + 0.65/0.20 GPU split.
#
#   srun --partition=gpu-test --gres=gpu:a100:1 --constraint=gpu80 \
#        --time=1:00:00 --mem=64G --cpus-per-task=4 --qos=gpu-test \
#        --job-name=plens7b bash <this-script> > logs/srun_7b.log 2>&1 &

set -euo pipefail

ROOT=/scratch/gpfs/ZHUANGL/sk7524
LF_VENV=$ROOT/LLaMA-Factory-AutoReviewer/.venv
WT_VENV=$ROOT/paperlens-training-and-inference/.venv
PP_ROOT=$ROOT/PaperLens/paperprep
REVIEW=$ROOT/tools/paperlens-reviewing
TEST_PDFS=$ROOT/LLaMA-Factory-AutoReviewer/test_pdfs

OUT_DIR=$REVIEW/logs/7b_test
mkdir -p "$OUT_DIR"
exec > >(tee -a "$OUT_DIR/run.log") 2>&1

PAPERPREP_PORT=18014
PAPERLENS_PORT=18012
REVIEW_PORT=18013
MINERU_PORT=40001

cat > "$OUT_DIR/serve_7b.yaml" <<EOF
server: {host: 127.0.0.1, port: $PAPERLENS_PORT}
model:
  # 7B (Qwen2.5-VL). RANKER.md §6.1 says ckpt-2618 was 7B's best step.
  ckpt_path: $ROOT/LLaMA-Factory-AutoReviewer/saves/final_sweep_v7_datasweepv3/final_data_sweep_v3/arxiv_train/small/arxiv_21k_vision/checkpoint-2618
  template: qwen2_vl
  cutoff_len: 24480
  enable_thinking: false
  dtype: bfloat16
vllm:
  tensor_parallel_size: 1
  pipeline_parallel_size: 1
  gpu_memory_utilization: 0.65    # 7B is ~2x 3B; paperprep drops to 0.20
  max_model_len: 24580
  disable_log_stats: true
scoring:
  positive_token: Accept
  negative_token: Reject
  decision_token_idx: 5
  max_new_tokens: 8
  image_max_pixels: 1003520
  image_min_pixels: 784
logging: {level: INFO}
EOF

echo "[`date +%T`] node=$(hostname)  out_dir=$OUT_DIR"
nvidia-smi -L | head -1

echo "[`date +%T`] launching paperprep serve (gpu-mem-util=0.20)"
env "PATH=$PP_ROOT/.venv/bin:$PATH" "$PP_ROOT/.venv/bin/python" \
    -m paper_anonymizer.paperprep.cli serve \
    --output-dir "$OUT_DIR/paperprep_work" \
    --host 127.0.0.1 --port "$PAPERPREP_PORT" \
    --mineru-port "$MINERU_PORT" \
    --gpu-memory-utilization 0.20 \
    > "$OUT_DIR/paperprep.log" 2>&1 &
PP_PID=$!
echo "[`date +%T`] paperprep PID=$PP_PID"

echo "[`date +%T`] launching paperlens serve (worktree venv, 7B ckpt)"
"$WT_VENV/bin/paperlens" serve \
    --config "$OUT_DIR/serve_7b.yaml" \
    --port "$PAPERLENS_PORT" \
    > "$OUT_DIR/paperlens.log" 2>&1 &
PL_PID=$!
echo "[`date +%T`] paperlens PID=$PL_PID"

echo "[`date +%T`] waiting for paperprep + paperlens /health ..."
for _ in {1..360}; do
  if curl -sf "http://127.0.0.1:$PAPERPREP_PORT/healthz" >/dev/null 2>&1 \
     && curl -sf "http://127.0.0.1:$PAPERLENS_PORT/health" >/dev/null 2>&1; then
    echo "[`date +%T`] both healthy"
    break
  fi
  sleep 5
done

echo "[`date +%T`] launching paperlensreview"
export PAPERPREP_SERVE_BASE_URL="http://127.0.0.1:$PAPERPREP_PORT"
export PAPERLENS_SERVE_BASE_URL="http://127.0.0.1:$PAPERLENS_PORT"
env "PYTHONPATH=$REVIEW/src" "$LF_VENV/bin/python" -m paperlensreview serve \
    --config "$REVIEW/configs/server.yaml" \
    --port "$REVIEW_PORT" \
    --skip_preflight \
    > "$OUT_DIR/paperlensreview.log" 2>&1 &
RV_PID=$!
sleep 8
echo "[`date +%T`] paperlensreview PID=$RV_PID"

trap 'echo "[cleanup] killing daemons"; kill $RV_PID $PL_PID $PP_PID 2>/dev/null; pkill -f mineru-vllm-server 2>/dev/null; exit' EXIT

echo "[`date +%T`] running 27-PDF batch"
BASE_URL=http://127.0.0.1:$REVIEW_PORT \
PYTHONPATH=$REVIEW/src \
OUT_PATH=$OUT_DIR/batch_results_7b.jsonl \
TEST_PDFS=$TEST_PDFS \
"$LF_VENV/bin/python" - <<'PY'
import json, os, time, urllib.request, sys
from pathlib import Path
BASE = os.environ['BASE_URL']
ROOT = Path(os.environ['TEST_PDFS'])
OUT  = Path(os.environ['OUT_PATH']); OUT.write_text('')
def http_get(url):
    with urllib.request.urlopen(url, timeout=10) as r: return json.loads(r.read())
def http_post(url, p: Path):
    boundary = '----p7b'+str(int(time.time()*1e6))
    body = f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{p.name}"\r\nContent-Type: application/pdf\r\n\r\n'.encode() + p.read_bytes() + f'\r\n--{boundary}--\r\n'.encode()
    req = urllib.request.Request(url, data=body, method='POST')
    req.add_header('Content-Type', f'multipart/form-data; boundary={boundary}')
    with urllib.request.urlopen(req, timeout=120) as r: return json.loads(r.read())
pdfs = sorted(ROOT.rglob('*.pdf'))
print(f'[batch] {len(pdfs)} pdfs', flush=True)
for i, p in enumerate(pdfs):
    rel = p.relative_to(ROOT)
    print(f'[{i+1}/{len(pdfs)}] {rel}', flush=True)
    try:
        resp = http_post(f'{BASE}/submit', p); jid = resp['job_id']
    except Exception as e:
        with OUT.open('a') as f: f.write(json.dumps({'pdf':str(rel),'state':'submit_failed','error':str(e)})+'\n'); continue
    deadline = time.time() + 480; s = None
    while time.time() < deadline:
        try: s = http_get(f'{BASE}/status/{jid}')
        except Exception: time.sleep(2); continue
        if s['state'] in ('done','error'): break
        time.sleep(3)
    rec = {'pdf': str(rel), 'state': s['state'] if s else 'timeout'}
    if s and s.get('result'):
        r = s['result']
        rec.update({'decision':r['decision'], 'p_accept':round(r['p_accept'],4),
                    'logp_acc':round(r['logp_accept'],3) if r.get('logp_accept') is not None else None,
                    'logp_rej':round(r['logp_reject'],3) if r.get('logp_reject') is not None else None,
                    'elapsed_s': r.get('paperprep_elapsed_s')})
    if s and s.get('error'): rec['error'] = s['error']
    with OUT.open('a') as f: f.write(json.dumps(rec)+'\n')
    print(f'  -> {rec.get("decision","?"):<6} p={rec.get("p_accept","-")}', flush=True)
print(f'\nDONE: {OUT}', flush=True)
PY

echo "[`date +%T`] batch done"
ls -la "$OUT_DIR/batch_results_7b.jsonl"
