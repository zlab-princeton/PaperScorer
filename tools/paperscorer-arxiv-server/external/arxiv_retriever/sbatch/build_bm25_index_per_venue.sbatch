#!/bin/bash
# Build a Pyserini Lucene (BM25) index over the SUBSET of arxiv papers we
# have MinerU + per-venue metadata for: union of arxiv_ids in
# arxiv_50_50_balanced_per_venue_text_wmetadata_filtered24480_{train,val,test}.
# That's ~83.6K papers vs 2.92M in the full arxiv index.
#
# Submit:
#   sbatch arxiv_retriever/sbatch/build_bm25_index_per_venue.sbatch
#
# Output: $OUT_DIR/ (lucene index, ~150-300 MB)
#
#SBATCH --job-name=bm25_pv_index
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/bm25_pv_index_%j.out
#SBATCH --error=logs/bm25_pv_index_%j.err

set -uo pipefail

WORKDIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$WORKDIR"
mkdir -p logs

SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/gpfs/ZHUANGL/sk7524}"
WIKI_JSONL="${SCRATCH_ROOT}/SkyRL/skyrl-train/data/searchr1_original/arxiv/arxiv_wikiformat.jsonl"
OUT_DIR="${OUT_DIR:-${SCRATCH_ROOT}/arxiv_bm25_per_venue_index}"
SUBSET_JSONL="${SUBSET_JSONL:-${SCRATCH_ROOT}/arxiv_bm25_per_venue_stage/arxiv_wikiformat_per_venue.jsonl}"
THREADS="${THREADS:-${SLURM_CPUS_PER_TASK:-16}}"

PV_FILES=(
    "${SCRATCH_ROOT}/LLaMA-Factory-AutoReviewer/data/arxiv_50_50_balanced_per_venue_text_wmetadata_filtered24480_train/data_part1.json"
    "${SCRATCH_ROOT}/LLaMA-Factory-AutoReviewer/data/arxiv_50_50_balanced_per_venue_text_wmetadata_filtered24480_train/data_part2.json"
    "${SCRATCH_ROOT}/LLaMA-Factory-AutoReviewer/data/arxiv_50_50_balanced_per_venue_text_wmetadata_filtered24480_validation/data.json"
    "${SCRATCH_ROOT}/LLaMA-Factory-AutoReviewer/data/arxiv_50_50_balanced_per_venue_text_wmetadata_filtered24480_test/data.json"
)

RETRIEVER_CONDA_ENV="${RETRIEVER_CONDA_ENV:-retriever}"

echo "=============================================="
echo "Job:        $SLURM_JOB_ID"
echo "Host:       $(hostname)"
echo "Wiki jsonl: $WIKI_JSONL"
echo "Subset:     $SUBSET_JSONL"
echo "Output:     $OUT_DIR"
echo "Threads:    $THREADS"
echo "=============================================="

mkdir -p "$(dirname "$SUBSET_JSONL")"
rm -f "$SUBSET_JSONL"
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

# Step 1: build subset jsonl (filter wikiformat to per_venue arxiv_ids)
echo "=== filtering wikiformat to per_venue arxiv_id subset ==="
python3 << PY
import json, time

PV_FILES = [
    "${PV_FILES[0]}",
    "${PV_FILES[1]}",
    "${PV_FILES[2]}",
    "${PV_FILES[3]}",
]
print("Loading per_venue arxiv_ids...", flush=True)
pv_ids = set()
for f in PV_FILES:
    t0 = time.time()
    with open(f) as fh:
        rows = json.load(fh)
    for r in rows:
        m = r.get("_metadata", {})
        a = m.get("arxiv_id")
        if a:
            pv_ids.add(a)
    print(f"  {f.rsplit('/',1)[-1]}: cumulative {len(pv_ids)} ids ({time.time()-t0:.1f}s)", flush=True)
print(f"Total per_venue arxiv_ids: {len(pv_ids)}", flush=True)

print("Filtering wikiformat...", flush=True)
t0 = time.time()
n_in = 0
n_total = 0
with open("${WIKI_JSONL}") as fin, open("${SUBSET_JSONL}", "w") as fout:
    for line in fin:
        n_total += 1
        d = json.loads(line)
        if d["id"] in pv_ids:
            fout.write(line)
            n_in += 1
print(f"Wrote {n_in} / {n_total} rows to ${SUBSET_JSONL} ({time.time()-t0:.1f}s)", flush=True)
print(f"Coverage of per_venue ids in wikiformat: {n_in}/{len(pv_ids)} ({100*n_in/len(pv_ids):.1f}%)", flush=True)
PY

ls -lh "$SUBSET_JSONL"

# Step 2: pyserini lucene index
set +u
module load anaconda3/2024.6
conda activate "$RETRIEVER_CONDA_ENV"
set -u

# Pyserini reads dir; the subset jsonl already lives in its own clean dir.
STAGE_DIR="$(dirname "$SUBSET_JSONL")"

echo ""
echo "=== running pyserini.index.lucene ($THREADS threads, per_venue subset) ==="
python -m pyserini.index.lucene \
    --collection JsonCollection \
    --input "$STAGE_DIR" \
    --index "$OUT_DIR" \
    --generator DefaultLuceneDocumentGenerator \
    --threads "$THREADS" \
    --storePositions \
    --storeDocvectors \
    --storeRaw

INDEX_RC=$?
echo ""
if [ "$INDEX_RC" -eq 0 ]; then
    echo "=== indexing complete ==="
    du -sh "$OUT_DIR"
else
    echo "ERROR: pyserini exited with $INDEX_RC"
    exit "$INDEX_RC"
fi

echo ""
echo "=== smoke: open + sample search ==="
python - <<PY
from pyserini.search.lucene import LuceneSearcher
s = LuceneSearcher("$OUT_DIR")
print(f"num_docs: {s.num_docs}")
hits = s.search("knowledge distillation language model compression", 5)
for i, h in enumerate(hits, 1):
    print(f"  hit {i}: docid={h.docid}  score={h.score:.3f}")
PY

echo ""
echo "Done."
