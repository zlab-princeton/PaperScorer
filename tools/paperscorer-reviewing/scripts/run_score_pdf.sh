#!/usr/bin/env bash
# Score a single pre-built (already-anonymized) PDF through paperprep-serve
# (type=pdf -> skip compile, mineru+normalize+filter+export) + paperlens-serve.
# Used to drop a cross-format reference point (the NeurIPS/COLM submission) next
# to the arXiv-template commit trajectory.
#
#   PDF_PATH=/abs/x.pdf LABEL=neurips srun --partition=gpu-test --gres=gpu:a100:1 \
#     --constraint=gpu80 --qos=gpu-test --time=1:00:00 --mem=64G --cpus-per-task=4 \
#     --job-name=scorepdf bash scripts/run_score_pdf.sh

set -uo pipefail

PDF_PATH="${PDF_PATH:?set PDF_PATH}"
LABEL="${LABEL:-pdf}"

ROOT=/scratch/gpfs/ZHUANGL/sk7524
LF_VENV=$ROOT/LLaMA-Factory-AutoReviewer/.venv
WT_VENV=$ROOT/paperlens-training-and-inference/.venv
PP_ROOT=$ROOT/PaperLens/paperprep
REVIEW=$ROOT/tools/paperlens-reviewing
SERVE_CFG=$ROOT/paperlens-training-and-inference/configs/serve.yaml

OUT_DIR=$REVIEW/logs/commit_track
mkdir -p "$OUT_DIR"
exec > >(tee -a "$OUT_DIR/run_${LABEL}.log") 2>&1

PAPERPREP_PORT=18030
PAPERLENS_PORT=18028
MINERU_PORT=40030

echo "[`date +%T`] score-pdf '$LABEL' node=$(hostname) pdf=$PDF_PATH"
nvidia-smi -L | head -1

env "PATH=$PP_ROOT/.venv/bin:$PATH" "$PP_ROOT/.venv/bin/python" \
    -m paper_anonymizer.paperprep.cli serve \
    --output-dir "$OUT_DIR/paperprep_work_${LABEL}" \
    --host 127.0.0.1 --port "$PAPERPREP_PORT" --mineru-port "$MINERU_PORT" \
    --gpu-memory-utilization 0.20 > "$OUT_DIR/paperprep_${LABEL}.log" 2>&1 &
PP_PID=$!
"$WT_VENV/bin/paperlens" serve --config "$SERVE_CFG" --port "$PAPERLENS_PORT" \
    > "$OUT_DIR/paperlens_${LABEL}.log" 2>&1 &
PL_PID=$!
trap 'kill $PL_PID $PP_PID 2>/dev/null; pkill -f mineru-vllm-server 2>/dev/null; exit' EXIT

echo "[`date +%T`] waiting for /health ..."
for _ in {1..360}; do
  curl -sf "http://127.0.0.1:$PAPERPREP_PORT/healthz" >/dev/null 2>&1 \
   && curl -sf "http://127.0.0.1:$PAPERLENS_PORT/health" >/dev/null 2>&1 && break
  sleep 5
done

PDF_PATH="$PDF_PATH" LABEL="$LABEL" \
PAPERPREP_URL="http://127.0.0.1:$PAPERPREP_PORT" \
PAPERLENS_URL="http://127.0.0.1:$PAPERLENS_PORT" \
OUT="$OUT_DIR/results_${LABEL}.jsonl" \
"$LF_VENV/bin/python" - <<'PY'
import json, os, requests
from pathlib import Path
pp, pl = os.environ["PAPERPREP_URL"], os.environ["PAPERLENS_URL"]
pdf, label, out = os.environ["PDF_PATH"], os.environ["LABEL"], Path(os.environ["OUT"])
body = requests.post(pp + "/prepare", json={"request_id": label,
        "papers": [{"id": label, "type": "pdf", "path": pdf}]}, timeout=900).json()
p0 = (body.get("papers") or [{}])[0]
rec = {"label": label, "pdf": pdf, "state": "pending"}
if p0.get("status") != "ok":
    rec.update(state="prep_failed", error=f"{p0.get('status')}: {p0.get('error')}")
else:
    sgp = body.get("sharegpt_vision_path")
    rows = json.loads(Path(sgp).read_text()) if sgp and Path(sgp).exists() else []
    if not rows:
        rec.update(state="no_sharegpt")
    else:
        row = dict(rows[0]); convs = list(row.get("conversations", []))
        if not any(c.get("from") == "gpt" for c in convs):
            convs.append({"from": "gpt", "value": "Outcome: \\boxed{Accept}"}); row["conversations"] = convs
        sc = requests.post(pl + "/score", json={"papers": [row]}, timeout=900).json()["scores"][0]
        pa = float(sc["p_accept"])
        rec.update(state="done", p_accept=round(pa, 4), decision="Accept" if pa >= 0.5 else "Reject",
                   logp_accept=sc.get("logp_accept"), logp_reject=sc.get("logp_reject"))
out.write_text(json.dumps(rec) + "\n")
print("RESULT:", json.dumps(rec))
PY

echo "[`date +%T`] DONE -> $OUT_DIR/results_${LABEL}.jsonl"
