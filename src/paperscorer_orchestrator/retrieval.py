"""`paperscorer setup --with-retrieval` — provision the search/rerank stack.

The reviewing path (`paperscorer run/serve`) needs none of this; retrieval is the
separate, heavier deployment (arxiv_retriever FAISS/BM25 server +
paperscorer-arxiv-server reranker + search UI). `setup --with-retrieval` runs:

  a. verify the per_venue vision dataset keys are registered + their data.json exist
  b. verify <data_root>/images_arxiv exists (page PNGs the reranker scores)
  c. clone the chosen retriever's index from HF (skonan/PaperScorer-arXiv-embeddings)
     into the HF cache, and ensure Qwen3-Embedding-0.6B is cached
  d. render a concrete arxiv_retriever config pointing at the resolved paths
  e. ensure the `retriever` conda env exists (faiss-gpu + pyserini; login-node
     build from arxiv_retriever/conda-env.yml)

`provision_retrieval(cfg)` runs a-e and prints a checklist; nothing here is
required by the core review flow, so failures are reported, not fatal.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from rich.console import Console

from .config import Config, ROOT

console = Console()

ARXIV_SERVER = ROOT / "tools" / "paperscorer-arxiv-server"
ARXIV_RETRIEVER = ARXIV_SERVER / "external" / "arxiv_retriever"
CONDA_ENV_NAME = "retriever"

# HF distribution (under the PaperScorer collection). Serve-minimal bundle:
# bm25 lucene index dir + qwen3 FAISS index + the per_venue corpus jsonl +
# precomputed p_accept caches (predictions parquets, ~28K seeded rows).
INDEX_HF_REPO = "skonan/PaperScorer-arXiv-embeddings"
EMBED_HF_REPO = "Qwen/Qwen3-Embedding-0.6B"

# The per_venue vision dataset keys retrieval/reranking depends on (labels +
# page images). All three splits make up the 80K per_venue corpus.
PER_VENUE_VISION_KEYS = [
    f"arxiv_50_50_balanced_per_venue_vision_wmetadata_filtered24480_{sp}"
    for sp in ("train", "validation", "test")
]

# Layout inside the HF index bundle (skonan/PaperScorer-arXiv-embeddings).
BUNDLE = {
    "bm25": "bm25_per_venue_index",                    # lucene index dir
    "faiss": "qwen3_06b_per_venue/qwen3_Flat.index",   # dense FAISS index
    "corpus": "corpus/arxiv_wikiformat_per_venue.jsonl",
}

# Where the generated retriever config lands (gitignored; launch_local.sh reads it).
GENERATED_CFG = ARXIV_SERVER / "configs" / "retriever.generated.yaml"


# ---------------------------------------------------------------------------
# (a)(b) data checks
# ---------------------------------------------------------------------------

def resolve_data_dir(cfg: Config) -> Path | None:
    """The dir holding dataset_info.json + images_arxiv. Accepts
    PAPERLENS_DATA_ROOT either as the data/ dir itself or as its parent.
    Returns None if PAPERLENS_DATA_ROOT is unset or neither layout matches.
    """
    if not cfg.cache.data_root:
        return None
    dr = Path(cfg.cache.data_root).expanduser()
    for c in (dr, dr / "data"):
        if (c / "dataset_info.json").is_file():
            return c
    return None


def check_data(cfg: Config) -> tuple[bool, Path | None, list[str]]:
    """(a) per_venue vision keys registered + data.json present; (b) images_arxiv.
    Returns (ok, data_dir, messages)."""
    data_dir = resolve_data_dir(cfg)
    if data_dir is None:
        return False, None, [
            "PAPERLENS_DATA_ROOT does not point at a dir containing dataset_info.json "
            "(checked <root> and <root>/data). Set it via `paperscorer setup` and run "
            "scripts/reconstruction.py to materialize the per_venue vision data."
        ]
    info = json.loads((data_dir / "dataset_info.json").read_text())
    msgs: list[str] = []
    ok = True
    for k in PER_VENUE_VISION_KEYS:
        if k not in info:
            ok = False
            msgs.append(f"missing dataset_info key: {k}")
            continue
        rel = info[k].get("file_name", f"{k}/data.json")
        if not (data_dir / rel).is_file():
            ok = False
            msgs.append(f"key {k} registered but data file missing: {data_dir / rel}")
    if not (data_dir / "images_arxiv").is_dir():
        ok = False
        msgs.append(f"images_arxiv/ missing under {data_dir}")
    return ok, data_dir, msgs


# ---------------------------------------------------------------------------
# (c) indexes + embedding model + predictions cache seed
# ---------------------------------------------------------------------------

# Which bundle pieces each retriever needs. bm25 needs the lucene index; qwen
# needs the dense FAISS index. Both need the per_venue corpus.
RETRIEVERS = ("qwen", "bm25")
_NEEDED = {"qwen": ("faiss", "corpus"), "bm25": ("bm25", "corpus")}
# HF allow_patterns per bundle key (fetch only what the chosen retriever needs).
_HF_PATTERNS = {
    "bm25": ["bm25_per_venue_index/*"],
    "faiss": ["qwen3_06b_per_venue/*"],
    "corpus": ["corpus/*"],
}


def resolve_index_paths(cfg: Config, which: str, *, allow_download: bool = True) -> tuple[dict | None, str]:
    """Resolve the index paths the chosen retriever (`which`) needs. Order:
      1. $PAPERLENS_RETRIEVAL_DIR (local override expecting the BUNDLE layout)
      2. HF bundle (snapshot_download into the HF cache, only the needed pieces)
    Returns (paths_or_None, source_label). bm25 pulls the BM25 index (+corpus);
    qwen pulls the FAISS index (+corpus).
    """
    needed = _NEEDED[which]
    override = os.environ.get("PAPERLENS_RETRIEVAL_DIR")
    if override:
        base = Path(override).expanduser()
        paths = {k: base / BUNDLE[k] for k in needed}
        if all(p.exists() for p in paths.values()):
            return paths, f"override:{base}"

    if allow_download:
        try:
            from huggingface_hub import snapshot_download
            allow = [pat for k in needed for pat in _HF_PATTERNS[k]]
            snap = Path(snapshot_download(INDEX_HF_REPO, repo_type="dataset", allow_patterns=allow))
            paths = {k: snap / BUNDLE[k] for k in needed}
            if all(p.exists() for p in paths.values()):
                return paths, f"HF:{INDEX_HF_REPO}"
        except Exception as e:
            console.print(f"  [dim]HF index download unavailable ({e})[/dim]")
    return None, "missing"


def resolve_embed_model(cfg: Config, *, allow_download: bool = True) -> Path | None:
    """Qwen3-Embedding-0.6B snapshot path (from HF cache; download if allowed)."""
    try:
        from huggingface_hub import snapshot_download
    except Exception:
        return None
    try:
        return Path(snapshot_download(EMBED_HF_REPO, local_files_only=True))
    except Exception:
        if not allow_download:
            return None
        try:
            return Path(snapshot_download(EMBED_HF_REPO))
        except Exception as e:
            console.print(f"  [yellow]could not fetch {EMBED_HF_REPO}: {e}[/yellow]")
            return None


def resolve_predictions_parquet(cfg: Config, *, allow_download: bool = True) -> Path | None:
    """Size-matched p_accept seed parquet from the HF bundle's ``cache/`` dir,
    snapshot_downloaded into the HF cache. Used at serve time to bootstrap the
    SQLite cache so the reranker hits ~28K precomputed scores out of the box.
    """
    if not allow_download:
        return None
    size = cfg.model.size  # '3b' | '7b'
    name = f"predictions_{size}.parquet"
    try:
        from huggingface_hub import snapshot_download
        snap = Path(snapshot_download(INDEX_HF_REPO, repo_type="dataset",
                                      allow_patterns=[f"cache/{name}"]))
        cand = snap / "cache" / name
        if cand.exists():
            return cand
    except Exception as e:
        console.print(f"  [dim]HF predictions download unavailable ({e})[/dim]")
    return None


def resolve_images_root(cfg: Config) -> Path | None:
    """Page-image root for the reranker (<data_dir>/images_arxiv)."""
    data_dir = resolve_data_dir(cfg)
    if data_dir is None:
        return None
    imgs = data_dir / "images_arxiv"
    return imgs if imgs.is_dir() else None


# ---------------------------------------------------------------------------
# (d) retriever config generation
# ---------------------------------------------------------------------------

def render_retriever_config(which: str, paths: dict, embed_model: Path | None, *,
                            out_path: Path = GENERATED_CFG, port: int = 8001) -> Path:
    """Write a concrete per_venue arxiv_retriever config for the chosen retriever
    (`which`), pointing at the resolved index/corpus(/model) paths. The
    arxiv_metadata_path is intentionally omitted -- the search UI doesn't use
    author/date/title filters, so skipping the 4.7G metadata keeps boot fast
    and the deployment portable.
    """
    header = ("# Generated by `paperscorer setup --with-retrieval` -- do NOT edit by hand\n"
              "# (re-run setup to regenerate).\n")
    if which == "bm25":
        yaml = header + f"""# BM25 (pyserini/lucene) per_venue retriever -- no neural encoder.
server:
  port: {port}
  faiss_gpu: false
  topk: 5

retriever:
  name: bm25
  model_path: ""
  pooling_method: mean
  query_max_length: 256
  use_fp16: false

index:
  path: {paths['bm25']}
  corpus_path: {paths['corpus']}
"""
    else:
        yaml = header + f"""# Dense per_venue retriever: Qwen3-Embedding-0.6B FAISS over the ~80K corpus.
server:
  port: {port}
  faiss_gpu: true
  start_awake: true      # load the embed model on GPU at boot (else it parks on CPU)
  topk: 5

retriever:
  name: qwen3
  model_path: {embed_model}
  pooling_method: last_token
  query_max_length: 256
  use_fp16: true
  task_description: "Given a query about academic research, retrieve relevant arXiv papers that address the topic"

index:
  path: {paths['faiss']}
  corpus_path: {paths['corpus']}
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml)
    return out_path


# ---------------------------------------------------------------------------
# (e) conda env
# ---------------------------------------------------------------------------

def _conda_bin() -> str | None:
    return shutil.which("conda")


def resolve_conda_envs_path(cfg: Config, *, dry_run: bool = False) -> str:
    """Where the `retriever` conda env lives (CONDA_ENVS_PATH). Resolution:
      1. live $CONDA_ENVS_PATH (used silently if set)
      2. saved cfg.cache.conda_envs_path
      3. derive <scratch root>/.conda from the parent of HF_HOME (else data_root),
         offered as the default; the user can override (skipped in dry-run)
      4. no default -> prompt with no default
    """
    live = os.environ.get("CONDA_ENVS_PATH")
    if live:
        return live
    if cfg.cache.conda_envs_path:
        return cfg.cache.conda_envs_path
    default = ""
    root_src = cfg.cache.hf_home or os.environ.get("HF_HOME") or cfg.cache.data_root
    if root_src:
        default = str(Path(root_src).expanduser().resolve().parent / ".conda")
    if dry_run:
        return default
    from rich.prompt import Prompt
    return Prompt.ask("  CONDA_ENVS_PATH (parent dir for the `retriever` env)",
                      default=default or None) or default


def conda_env_exists(envs_path: str = "", name: str = CONDA_ENV_NAME) -> bool:
    # Direct check at <envs_path>/<name> is authoritative and doesn't depend on
    # CONDA_ENVS_PATH being active in `conda env list`.
    if envs_path and (Path(envs_path).expanduser() / name / "bin" / "python").exists():
        return True
    conda = _conda_bin()
    if not conda:
        return False
    env = dict(os.environ)
    if envs_path:
        env["CONDA_ENVS_PATH"] = envs_path
    r = subprocess.run([conda, "env", "list"], capture_output=True, text=True, env=env)
    for line in r.stdout.splitlines():
        parts = line.split()
        if parts and (parts[0] == name or parts[-1].rstrip("/").endswith("/" + name)):
            return True
    return False


def conda_env_spec() -> Path | None:
    """conda-env.yml inside the arxiv_retriever submodule (the canonical spec)."""
    spec = ARXIV_RETRIEVER / "conda-env.yml"
    return spec if spec.is_file() else None


def ensure_conda_env(envs_path: str, *, dry_run: bool = False) -> bool:
    """Build the `retriever` conda env from conda-env.yml on the login node,
    landing it in CONDA_ENVS_PATH=<envs_path>. Returns True if the env exists
    (or would, in dry-run) afterward.
    """
    loc = envs_path or "conda default"
    if conda_env_exists(envs_path):
        console.print(f"  [green]✓[/green] conda env '{CONDA_ENV_NAME}' exists ([dim]{loc}[/dim])")
        return True
    spec = conda_env_spec()
    if spec is None:
        console.print("  [yellow]conda-env.yml not found[/yellow] — cannot build the retriever env")
        return False
    envline = f"CONDA_ENVS_PATH={envs_path} " if envs_path else ""
    conda = _conda_bin()
    if not conda:
        console.print("  [yellow]conda not on PATH[/yellow] — load anaconda, then:\n"
                      f"    [cyan]{envline}conda env create -f {spec}[/cyan]")
        return False
    if dry_run:
        console.print(f"  [dim](dry-run) {envline}conda env create -f {spec}  (cwd={spec.parent})[/dim]")
        return True
    console.print(f"  building conda env '{CONDA_ENV_NAME}' at [dim]{loc}[/dim] from {spec} (login node) ...")
    # cwd = spec dir so the `-e .` in conda-env.yml installs *this* arxiv_retriever
    # checkout. CONDA_ENVS_PATH lands the env next to the user's other envs
    # instead of the home-quota default.
    env = dict(os.environ)
    if envs_path:
        env["CONDA_ENVS_PATH"] = envs_path
    r = subprocess.run([conda, "env", "create", "-f", str(spec)], cwd=str(spec.parent), env=env)
    ok = r.returncode == 0
    console.print(f"  {'[green]✓[/green]' if ok else f'[red]✗ rc={r.returncode}[/red]'} conda env '{CONDA_ENV_NAME}'")
    return ok


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------

def provision_retrieval(cfg: Config, *, dry_run: bool = False) -> bool:
    """Run a-e and print a checklist. Returns True if the stack looks ready to
    launch (indexes + config + conda env resolved)."""
    console.rule("[bold]Retrieval stack (search + rerank)")

    # (a)(b) data
    ok_data, data_dir, msgs = check_data(cfg)
    if ok_data:
        console.print(f"  [green]✓[/green] per_venue vision data + images_arxiv ([dim]{data_dir}[/dim])")
    else:
        console.print("  [yellow]✗ data not ready:[/yellow]")
        for m in msgs:
            console.print(f"      • {m}")

    # (c) retriever (chosen in the setup form -> cfg.retrieval.retriever) -> indexes
    which = cfg.retrieval.retriever if cfg.retrieval.retriever in RETRIEVERS else "qwen"
    console.print(f"  retriever: [bold]{which}[/bold]"
                  + ("  [dim](dense; index + model)[/dim]" if which == "qwen"
                     else "  [dim](lucene; index only)[/dim]"))
    paths, src = resolve_index_paths(cfg, which, allow_download=not dry_run)
    if paths:
        console.print(f"  [green]✓[/green] {which} per_venue index ([dim]{src}[/dim])")
        for k in _NEEDED[which]:
            console.print(f"      • {k}: [dim]{paths[k]}[/dim]")
    else:
        console.print(f"  [yellow]✗ {which} per_venue index not found[/yellow] "
                      f"(no $PAPERLENS_RETRIEVAL_DIR override, not in HF cache from {INDEX_HF_REPO})")

    embed = None
    if which == "qwen":
        embed = resolve_embed_model(cfg, allow_download=not dry_run)
        if embed:
            console.print(f"  [green]✓[/green] embed model Qwen3-Embedding-0.6B ([dim]{embed}[/dim])")
        else:
            console.print(f"  [yellow]✗ embed model not cached[/yellow] ({EMBED_HF_REPO})")
    embed_ok = (which == "bm25") or (embed is not None)

    # (d) config
    cfg_path = None
    if paths and embed_ok and not dry_run:
        cfg_path = render_retriever_config(which, paths, embed)
        console.print(f"  [green]✓[/green] wrote retriever config [dim]{cfg_path}[/dim]")
    elif paths and embed_ok and dry_run:
        console.print(f"  [dim](dry-run) would write {which} retriever config -> {GENERATED_CFG}[/dim]")

    # (e) conda env
    envs_path = resolve_conda_envs_path(cfg, dry_run=dry_run)
    if not dry_run and envs_path and cfg.cache.conda_envs_path != envs_path:
        cfg.cache.conda_envs_path = envs_path
        try:
            cfg.save()
        except Exception:
            pass
    ok_conda = ensure_conda_env(envs_path, dry_run=dry_run)

    # how to launch
    ready = bool(paths and embed_ok and ok_conda)
    console.print()
    if ready and cfg_path:
        prefix = "" if (os.environ.get("CONDA_ENVS_PATH") or not envs_path) else f"CONDA_ENVS_PATH={envs_path} "
        console.print("  [bold]launch:[/bold] from tools/paperscorer-arxiv-server/\n"
                      f"    [cyan]{prefix}RETRIEVER_CFG={cfg_path} bash scripts/launch_local.sh[/cyan]")
    elif ready and dry_run:
        console.print("  [dim](dry-run) all checks pass — re-run without --dry-run to write the "
                      "config, then launch.[/dim]")
    else:
        console.print("  [dim]retrieval stack not fully ready — see the unchecked items above.[/dim]")
    console.print()
    return ready
