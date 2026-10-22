# Production batch run — scoring papers (incl. git history) across many PaperLens models

This documents the **offline, batch** side of `paperlens-reviewing` — distinct from
the single-paper FastAPI UI in [`README.md`](README.md). It scores a fixed set of
papers — **including every commit of a paper's git repo** plus venue renders and a
held-out test set — with **multiple PaperLens checkpoints**, then renders the
trajectory plots, advisor maps, and grid heatmaps.

Everything lands under `logs/grid/`.

---

## Why two stages

`paperprep` (MinerU OCR) and `paperlens serve` (vLLM) both want the whole GPU;
co-residency OOMs. So prep and scoring are fully decoupled:

```
STAGE A  prep ONCE          STAGE B  score N models           STAGE C  plot
─────────────────────       ──────────────────────────       ─────────────────
paperprep-serve (GPU)       paperlens serve (GPU), per ckpt   matplotlib (CPU)
 latex_dir|pdf → OCR →       cached <modality>.json ──POST──▶  scores/*.jsonl →
 normalize → filter →        /score → p_accept                 heatmaps / maps /
 ShareGPT → CACHE                                              trajectories
 logs/grid/prepped/<pid>/    logs/grid/scores/<model>.jsonl
```

Prep is expensive (compile + OCR); scoring is cheap. The cache is built once and
reused by every model, so adding a model never re-preps.

---

## The papers

`build_prep_cache.py` assembles a stable, ordered paper list from three sources:

| source | flag | type | notes |
|---|---|---|---|
| git commits of a paper repo | `--repo <dir> --from-commit <sha>` | `latex_dir` | each commit `git archive`'d, compiled + anonymized by paperprep |
| a test-PDF tree | `--test-pdfs-root <dir>` | `pdf` | grouped by first subdir (e.g. `anon_gt/`, `iclr_best_paper/`) |
| extra one-off PDFs | `--extra-pdf label=/abs/x.pdf` | `pdf` | e.g. `neurips=`, `colm=` venue renders |

Each paper caches to `logs/grid/prepped/<paper_id>/`:
`meta.json` (group, order, date, subject, state), `text.json`, `vision.json`
(self-contained: page PNGs copied into `page_images/`), or a `FAILED` touch file.
The cache is **idempotent** — re-runs skip `state==ok` papers, retry `FAILED`.

---

## The models

`scripts/grid_models.tsv` — one model per line: `label⟶ckpt_relpath⟶template⟶modality`
(ckpt paths are relative to `LLaMA-Factory-AutoReviewer/saves/`). The canonical grid
is the 10 checkpoints: {arxiv-small, arxiv-large, iclr} × {text, vision}, 3B + 7B
(no 3B-text for arxiv-small/large). Epoch rule: 7B → ep2, 3B → ep4.

---

## Run it

All GPU jobs use Slurm `gpu-test` (1 h cap, **3 concurrent jobs**, `--constraint=gpu80`).

### Stage A — prep cache (paperprep-serve only)

```bash
cd /scratch/gpfs/ZHUANGL/sk7524/tools/paperlens-reviewing
# slice-parallel (CRC32 hash of paper_id % NUM_SLICES); 1 slice is fine for small sets
for s in 0 1 2; do
  SLICE_INDEX=$s NUM_SLICES=3 sbatch --partition=gpu-test --gres=gpu:a100:1 \
    --constraint=gpu80 --qos=gpu-test --time=1:00:00 --mem=64G --cpus-per-task=4 \
    --job-name=prep$s --export=ALL,SLICE_INDEX=$s,NUM_SLICES=3 scripts/run_prep_cache.sh
done
```

`run_prep_cache.sh` pins the paper repo, test-PDF root, and the `neurips=`/`colm=`
extra PDFs; edit those vars to change the paper set. After all slices finish, every
`prepped/<pid>/meta.json` should be `state: ok` (3 early commits don't compile and
1 anon_gt paper isn't anonymizable — those stay `FAILED` and are skipped downstream).

### Stage B — score N models (paperlens serve only)

`run_score_model.sh` reads `MODEL_LINES` (1-based line numbers into `grid_models.tsv`),
writes a per-model `serve.yaml` (swaps `ckpt_path`+`template`), launches `paperlens serve`,
scores the whole cache, kills the serve, repeats. **Submit one model per job** (avoid
comma-valued `MODEL_LINES` — see gotchas):

```bash
for ln in $(seq 1 10); do
  sbatch --partition=gpu-test --gres=gpu:a100:1 --constraint=gpu80 --qos=gpu-test \
    --time=1:00:00 --mem=64G --cpus-per-task=4 --job-name=score$ln \
    --export=ALL,MODEL_LINES=$ln scripts/run_score_model.sh
done
```

`score_from_cache.py` **truncates and rewrites** its `scores/<label>.jsonl` each run,
re-scoring the whole cache — so to add one paper, re-prep it (cached papers skip) then
re-run the models; there's no incremental append.

### Stage C — assemble + plot (CPU, no GPU)

```bash
PY=/scratch/gpfs/ZHUANGL/sk7524/LLaMA-Factory-AutoReviewer/.venv/bin/python
$PY scripts/assemble_grid.py --scores-dir logs/grid/scores \
    --models-tsv scripts/grid_models.tsv --out-dir logs/grid   # heatmaps + 10-model trajectory + summary.txt
$PY scripts/make_advisor_maps.py        # 3 landscape maps (4× 7B models): golden / iclr / our-lab
$PY scripts/plot_paperlens_versions.py  # PaperLens lifetime: COLM → NeurIPS → arXiv commits, 4 models
```

Outputs (`logs/grid/`): `scores_long.csv`, `summary.txt`, `heatmap_<group>.png`,
`commit_trajectory_allmodels.png`, and `maps/{map1_golden,map2_iclr,map3_ourlab,paperlens_versions}.png`.

### Single-model commit trajectory (lighter path)

For just one model over a repo's commits (no full grid): `run_commit_track.sh` +
`score_commits.py` → `plot_commit_trajectory.py`. `run_score_pdf.sh` scores a lone PDF.

---

## Gotchas (all hit in practice)

- **`sbatch --export` comma bug**: `--export=ALL,MODEL_LINES=1,2` parses the comma as a
  KEY=VAL delimiter → `MODEL_LINES` becomes just `1`. Submit one value per job.
- **`/health` races the engine**: paperlens `/health` returns 200 before vLLM finishes
  loading weights; if the node has stale GPU memory the engine then dies and every
  `/score` 500s while the launcher already started scoring. Symptom: a model with
  `err=N` on every paper, serve log stuck at "Loading safetensors shards: 2/7". Fix:
  re-run with `--exclude=<bad_node>` (we hit a persistent zombie on `della-l09g7`).
- **Inode hygiene**: MinerU dumps hundreds of crop JPGs per paper into the shared
  ZHUANGL quota (~175 M cap, chronically near full). `build_prep_cache.py` copies only
  the page PNGs into the cache then `rmtree`s the paperprep output subtree per paper.
- **Commit pids are 8-char**: `paper_id = "commit_" + sha[:8]`. `git log %h` is 7 — use
  `%H` and slice `[:8]` when matching (see `plot_paperlens_versions.py`).
- **Order commits by real git timestamp**, not the cache's day-granularity `date`:
  same-day commits otherwise sort arbitrarily. `plot_paperlens_versions.py` re-derives
  order from `git log --format=%cI` and caps the timeline at a named commit (`CAP_SHA`).
- **gpu-test only**: `--partition=gpu` is disallowed for srun/sbatch here; 3 jobs max
  concurrent, so 10 models run in ~4 waves.

---

## Map / plot conventions (advisor deliverables)

- `make_advisor_maps.py`: papers × 4 (7B) models. Mark **before** the title: `*`=in
  ArXiv-L train, `^`=in ICLR train, `*^`=both (training membership = contamination flag).
  A 2nd label line `(<Accept|Reject> @ VENUE'YY)` is the ground-truth outcome in our
  dataset (train OR test/val), so held-out test papers show outcomes without a mark.
  Whitespace columns carry no tick (true separation); brackets label sub-groups.
- `plot_paperlens_versions.py`: one chronological line per model across the paper's
  lifetime (venue submissions, then commit history), x-labels `MM/DD\n(message)`.
