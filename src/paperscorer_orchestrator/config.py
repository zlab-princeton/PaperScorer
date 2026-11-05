"""Repo-local config for the PaperScorer orchestrator CLI.

Stored at <PaperScorer-root>/.paperscorer.toml. Flat sections of string/bool
values, so we hand-serialize TOML (no tomli_w dependency). Read uses the
stdlib tomllib (py3.11+).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
from typing import Any

try:
    import tomllib  # py3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


def _looks_like_root(p: Path) -> bool:
    """A PaperScorer checkout has the submodule dirs (or a .gitmodules naming them)."""
    if (p / "paperprep").is_dir() and (p / "paperscorer-training-and-inference").is_dir():
        return True
    gm = p / ".gitmodules"
    return gm.is_file() and "paperscorer-training-and-inference" in gm.read_text()


def _discover_root() -> Path:
    """Find the PaperScorer checkout. Resolution order:
      1. $PAPERLENS_ROOT
      2. walk up from CWD looking for the submodule layout
      3. the install-relative parent (works only for an editable/in-tree install)
      4. CWD as a last resort
    `uv tool install` copies the package out of the source tree, so the
    install-relative path is NOT reliable -- CWD discovery is the primary path.
    """
    env = os.environ.get("PAPERLENS_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    cwd = Path.cwd().resolve()
    for cand in (cwd, *cwd.parents):
        if _looks_like_root(cand):
            return cand
    pkg_parent = Path(__file__).resolve().parents[2]
    if _looks_like_root(pkg_parent):
        return pkg_parent
    return cwd


ROOT = _discover_root()
CONFIG_PATH = ROOT / ".paperscorer.toml"
# Name used before the PaperScorer rename. Still read when the new file is
# absent, so existing checkouts keep their settings; `save` writes the new name.
LEGACY_CONFIG_PATH = ROOT / ".paperlens.toml"

MODALITIES = ("text", "vision")
SIZES = ("3b", "7b")
DOMAINS = ("arxiv", "iclr")


def model_repo(size: str, modality: str, domain: str) -> str:
    """HF repo id for a (size, modality, domain) triple. The ICLR models are
    published as ...-openreview-iclr; HF resolves repo ids case-insensitively."""
    suffix = "openreview-iclr" if domain == "iclr" else domain
    return f"skonan/paperscorer-{size}-{modality}-{suffix}"


def legacy_model_repo(size: str, modality: str, domain: str) -> str:
    """Pre-rename id, still the HF cache key for models downloaded before it."""
    return f"skonan/paperlens-{size}-{modality}-{domain}"


@dataclass
class CacheCfg:
    hf_home: str = ""                 # HF_HOME (model snapshot cache)
    uv_cache: str = ""                # UV_CACHE_DIR
    vllm_cache: str = ""              # VLLM_CACHE_ROOT
    triton_cache: str = ""            # TRITON_CACHE_DIR
    data_root: str = ""               # PAPERLENS_DATA_ROOT (reconstructed datasets / images)
    conda_envs_path: str = ""         # CONDA_ENVS_PATH (parent dir for the `retriever` conda env)


def parse_repo(repo: str) -> tuple[str, str, str]:
    """skonan/paperscorer-7b-vision-arxiv -> ('7b','vision','arxiv').
    Also accepts pre-rename skonan/paperlens-* ids and the -openreview-iclr suffix."""
    parts = repo.split("/")[-1].lower().split("-")  # [paperscorer, 7b, vision, (openreview-)iclr]
    return parts[1], parts[2], parts[-1]


@dataclass
class ModelCfg:
    """Multiple models may be selected (downloaded); one is `active` for
    run/serve (which can only host a single model at a time)."""
    repos: list[str] = field(default_factory=lambda: ["skonan/paperscorer-7b-vision-arxiv"])
    active: str = "skonan/paperscorer-7b-vision-arxiv"

    @property
    def repo(self) -> str:
        """The active model repo (used by run/serve/dashboard)."""
        if self.active and self.active in self.repos:
            return self.active
        return self.repos[0] if self.repos else "skonan/paperscorer-7b-vision-arxiv"

    @property
    def size(self) -> str:      return parse_repo(self.repo)[0]
    @property
    def modality(self) -> str:  return parse_repo(self.repo)[1]
    @property
    def domain(self) -> str:    return parse_repo(self.repo)[2]


@dataclass
class GpuCfg:
    mode: str = "slurm"               # local | slurm  (slurm is the default deployment)
    partition: str = ""               # slurm only — REQUIRED; set in wizard (--princeton fills gpu-test)
    time: str = "1:00:00"             # slurm walltime
    mem: str = "64G"                  # slurm host RAM (model load needs CPU RAM headroom)
    constraint: str = ""              # legacy; constraints now live in extra_args
    extra_args: str = ""              # srun args (e.g. --account=…, --constraint=…); cluster-specific


@dataclass
class RetrievalCfg:
    retriever: str = "qwen"           # qwen | bm25  (the --with-retrieval default)


@dataclass
class PaperprepCfg:
    # Bin dir containing pdflatex/latexmk for LaTeX-dir reviews. Empty = rely
    # on whatever's on $PATH when paperprep-serve starts. Set this if TeX Live
    # lives somewhere non-standard (no system install, shared cluster mount,
    # or pinning to a specific TeX Live version).
    texlive_bin: str = ""


@dataclass
class Config:
    cache: CacheCfg = field(default_factory=CacheCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    gpu: GpuCfg = field(default_factory=GpuCfg)
    retrieval: RetrievalCfg = field(default_factory=RetrievalCfg)
    paperprep: PaperprepCfg = field(default_factory=PaperprepCfg)
    env: dict[str, str] = field(default_factory=dict)   # extra env vars to export

    # -------- (de)serialize --------
    @classmethod
    def load(cls, path: Path = CONFIG_PATH) -> "Config":
        if not path.exists():
            if path == CONFIG_PATH and LEGACY_CONFIG_PATH.exists():
                path = LEGACY_CONFIG_PATH
            else:
                return cls()
        data = tomllib.loads(path.read_text())
        c = cls()

        def _apply(obj, d):
            # Only set real dataclass fields (some are read-only @property now,
            # e.g. model.size); silently ignore legacy/unknown keys.
            valid = {f.name for f in fields(obj)}
            for k, v in (d or {}).items():
                if k in valid:
                    setattr(obj, k, v)

        _apply(c.cache, data.get("cache"))
        _apply(c.model, data.get("model"))
        _apply(c.gpu, data.get("gpu"))
        _apply(c.retrieval, data.get("retrieval"))
        _apply(c.paperprep, data.get("paperprep"))
        # Back-compat: a pre-multiselect [model] with size/modality/domain.
        m = data.get("model") or {}
        if "repos" not in m and {"size", "modality", "domain"} <= set(m):
            repo = model_repo(m["size"], m["modality"], m["domain"])
            c.model.repos = [repo]
            c.model.active = repo
        # Configs saved before the PaperScorer rename hold skonan/paperlens-* ids.
        c.model.repos = [model_repo(*parse_repo(r)) for r in c.model.repos]
        if c.model.active:
            c.model.active = model_repo(*parse_repo(c.model.active))
        c.env = dict(data.get("env") or {})
        return c

    def save(self, path: Path = CONFIG_PATH) -> None:
        path.write_text(self._to_toml())

    def _to_toml(self) -> str:
        def esc(v: Any) -> str:
            if isinstance(v, bool):
                return "true" if v else "false"
            if isinstance(v, (list, tuple)):
                return "[" + ", ".join(esc(x) for x in v) + "]"
            return '"' + str(v).replace("\\", "\\\\").replace('"', '\\"') + '"'
        lines = ["# PaperScorer orchestrator config. Edit via `paperscorer setup`,",
                 "# `paperscorer model`, `paperscorer gpu` -- or by hand.", ""]
        lines.append("[cache]")
        for k, v in asdict(self.cache).items():
            lines.append(f"{k} = {esc(v)}")
        lines.append("")
        lines.append("[model]")
        for k, v in asdict(self.model).items():
            lines.append(f"{k} = {esc(v)}")
        lines.append("")
        lines.append("[gpu]")
        for k, v in asdict(self.gpu).items():
            lines.append(f"{k} = {esc(v)}")
        lines.append("")
        lines.append("[retrieval]")
        for k, v in asdict(self.retrieval).items():
            lines.append(f"{k} = {esc(v)}")
        lines.append("")
        lines.append("[paperprep]")
        for k, v in asdict(self.paperprep).items():
            lines.append(f"{k} = {esc(v)}")
        lines.append("")
        lines.append("[env]")
        for k, v in self.env.items():
            lines.append(f"{k} = {esc(v)}")
        lines.append("")
        return "\n".join(lines)

    def export_env(self) -> dict[str, str]:
        """The env dict to inject when launching child services."""
        # Start from the live environment, then override only with non-empty
        # config values. An unset config field leaves any existing $VAR intact
        # (we always prefer an env var the user already has set).
        e = dict(os.environ)
        if self.cache.hf_home:
            e["HF_HOME"] = self.cache.hf_home
            e["HF_HUB_CACHE"] = str(Path(self.cache.hf_home) / "hub")
        if self.cache.uv_cache:
            e["UV_CACHE_DIR"] = self.cache.uv_cache
        if self.cache.vllm_cache:
            e["VLLM_CACHE_ROOT"] = self.cache.vllm_cache
        if self.cache.triton_cache:
            e["TRITON_CACHE_DIR"] = self.cache.triton_cache
        if self.cache.data_root:
            e["PAPERLENS_DATA_ROOT"] = self.cache.data_root
            e["PAPERLENS_IMAGES_ROOT"] = str(Path(self.cache.data_root) / "images_arxiv")
        if self.cache.conda_envs_path:
            e["CONDA_ENVS_PATH"] = self.cache.conda_envs_path
        e.update(self.env)
        return e
