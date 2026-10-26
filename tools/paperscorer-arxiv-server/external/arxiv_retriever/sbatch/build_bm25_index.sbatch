#!/bin/bash
# Build a Pyserini Lucene (BM25) index over the arxiv corpus
# (`arxiv_wikiformat.jsonl`, ~2.92M papers).
#
# CPU-only — no GPU needed for BM25 indexing. Uses pyserini's
# `pyserini.index.lucene` direct-to-Java indexer.
#
# Submit:
#   sbatch arxiv_retriever/sbatch/build_bm25_index.sbatch
#
# Output: $OUT_DIR/ (lucene index files: segments_*, _0.cfe, etc.)
#
#SBATCH --job-name=bm25_index
##SBATCH --partition=<cpu-partition>   # optional: uncomment and set for your cluster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/bm25_index_%j.out
#SBATCH --error=logs/bm25_index_%j.err

set -uo pipefail

WORKDIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$WORKDIR"
mkdir -p logs

# Root that holds the inputs/outputs below; set SCRATCH_ROOT to your data dir.
SCRATCH_ROOT="${SCRATCH_ROOT:-${WORKDIR}/data}"
CORPUS_JSONL="${CORPUS_JSONL:-${SCRATCH_ROOT}/SkyRL/skyrl-train/data/searchr1_original/arxiv/arxiv_wikiformat.jsonl}"
OUT_DIR="${OUT_DIR:-${SCRATCH_ROOT}/arxiv_bm25_index}"
THREADS="${THREADS:-${SLURM_CPUS_PER_TASK:-16}}"

# Pyserini reads ALL files under --input as documents to index. The jsonl
# lives in a dir with siblings (metadata snapshot, helper scripts, dense
# index). Stage a clean dir containing only the wikiformat jsonl via symlink.
STAGE_DIR="${STAGE_DIR:-${SCRATCH_ROOT}/arxiv_bm25_stage}"

RETRIEVER_CONDA_ENV="${RETRIEVER_CONDA_ENV:-retriever}"

echo "=============================================="
echo "Job:        $SLURM_JOB_ID"
echo "Host:       $(hostname)"
echo "Corpus:     $CORPUS_JSONL"
echo "Stage dir:  $STAGE_DIR"
echo "Output:     $OUT_DIR"
echo "Threads:    $THREADS"
echo "=============================================="
test -f "$CORPUS_JSONL" || { echo "ERROR: corpus jsonl missing: $CORPUS_JSONL"; exit 1; }

# Stage: clean dir with only the wikiformat jsonl
rm -rf "$STAGE_DIR"
mkdir -p "$STAGE_DIR"
ln -s "$CORPUS_JSONL" "$STAGE_DIR/$(basename "$CORPUS_JSONL")"
echo "staged: $(ls -la "$STAGE_DIR")"

# Output: clean target
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

# Activate conda env (has pyserini + java)
set +u
module load anaconda3/2024.6
conda activate "$RETRIEVER_CONDA_ENV"
set -u

# Sanity: pyserini import + java
python -c "from pyserini.index.lucene import LuceneIndexer; print('pyserini OK')" || {
    echo "ERROR: pyserini import failed in env $RETRIEVER_CONDA_ENV"; exit 1;
}
which java && java -version 2>&1 | head -3

echo ""
echo "=== running pyserini.index.lucene (24 threads, ~2.9M docs) ==="
# Default-ish BM25 tokenization (DefaultLuceneDocumentGenerator: lowercase,
# Porter stemmer, English stopwords). Stores docids from the `id` field.
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
    ls -lah "$OUT_DIR"
    du -sh "$OUT_DIR"
else
    echo "ERROR: pyserini exited with $INDEX_RC"
    exit "$INDEX_RC"
fi

# Cleanup stage dir (just symlink, but tidy)
rm -rf "$STAGE_DIR"

echo ""
echo "=== smoke: open index + sample search ==="
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
