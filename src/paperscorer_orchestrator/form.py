"""Inline setup walkthrough for `paperscorer setup` (claude-code style).

A guided sequence of arrow-navigable menus (via `questionary`). Every step is
a menu with a `continue ›` and a `‹ back` row — except the first step, which
has nothing to go back to. You never type a back sentinel; you arrow to the
row you want. Ctrl-C aborts.

Steps: caches → data root → models → (active) → gpu mode → (slurm settings).

Falls back to the rich numbered-menu pages when there's no TTY / questionary
is missing. run_setup_form(cfg) -> Config | "__cancel__" | None
"""
from __future__ import annotations

import getpass
import glob
import os
import sys
from pathlib import Path

from .config import Config, MODALITIES, SIZES, DOMAINS, ROOT, model_repo

BACK = "__back__"
CANCEL = "__cancel__"
CONTINUE = "continue ›"
BACKROW = "‹ back"

# (config field, env var, subdir under the cache parent)
CACHE_VARS = [
    ("uv_cache",     "UV_CACHE_DIR",     "uv"),
    ("hf_home",      "HF_HOME",          "hf"),
    ("vllm_cache",   "VLLM_CACHE_ROOT",  "vllm"),
    ("triton_cache", "TRITON_CACHE_DIR", "triton"),
]


def _has_tty() -> bool:
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except Exception:
        return False


def _qstyle():
    from questionary import Style
    return Style([
        ("qmark", "fg:#1f6feb bold"), ("question", "bold"),
        ("pointer", "fg:#1f6feb bold"), ("highlighted", "fg:#1f6feb bold"),
        ("selected", "fg:#1a7f37"), ("answer", "fg:#1a7f37 bold"),
        ("instruction", "fg:#767676"),
    ])


def confirm_menu(msg: str, *, yes: str = "yes ›", no: str = "skip",
                 default_yes: bool = True) -> bool:
    """Arrow-navigable yes/skip gate matching the setup-form widgets, so every
    decision point uses the same select UX. Falls back to a rich y/n confirm
    when there's no TTY / questionary is unavailable."""
    if _has_tty():
        try:
            import questionary
            r = questionary.select(
                msg, choices=[yes, no], style=_qstyle(),
                default=(yes if default_yes else no),
                instruction="(↑↓ then Enter)",
            ).ask()
            if r is not None:
                return r == yes
        except Exception:
            pass
    from rich.prompt import Confirm
    return bool(Confirm.ask(f"  {msg}", default=default_yes))


def _default_parent() -> str:
    """Writable per-user scratch if we can find one, else /scratch/gpfs, else ~."""
    user = os.environ.get("USER") or getpass.getuser()
    for p in sorted(glob.glob(f"/scratch/gpfs/*/{user}")):
        if Path(p).is_dir():
            return p
    if Path("/scratch/gpfs").is_dir():
        return "/scratch/gpfs"
    return str(Path.home())


def _has_local_gpu() -> bool:
    """True iff nvidia-smi is present and lists at least one GPU here."""
    import shutil
    import subprocess
    if not shutil.which("nvidia-smi"):
        return False
    try:
        r = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=15)
        return r.returncode == 0 and "GPU" in r.stdout
    except Exception:
        return False


def run_setup_form(cfg: Config, *, with_retrieval: bool = False):
    if not _has_tty():
        return None
    try:
        import questionary
        from questionary import Style, Choice, Separator
    except Exception:
        return None

    # Non-selectable divider that visually splits the nav rows (continue / back)
    # from the editable field rows above them.
    def SEP():
        return Separator("─" * 36)

    style = Style([
        ("qmark", "fg:#1f6feb bold"), ("question", "bold"),
        ("pointer", "fg:#1f6feb bold"), ("highlighted", "fg:#1f6feb bold"),
        ("selected", "fg:#1a7f37"), ("answer", "fg:#1a7f37 bold"),
        ("instruction", "fg:#767676"),
    ])

    def _menu(msg, rows, default=None):
        r = questionary.select(
            msg, choices=rows, style=style,
            default=default if default in rows else None,
            instruction="(↑↓ then Enter)",
        ).ask()
        return CANCEL if r is None else r

    def _text(msg, default):
        r = questionary.text(msg, default=default or "", style=style,
                             instruction="(Enter to accept)").ask()
        return CANCEL if r is None else r

    # ---- working state, seeded from cfg + environment -------------------
    parent = _default_parent()
    st: dict = {
        "cache_parent": parent,
        # per cache var: prefer an already-set $VAR, then saved config, then
        # the parent-derived default. (User asked: always reuse env if set.)
        **{key: (os.environ.get(envname) or getattr(cfg.cache, key) or str(Path(parent) / sub))
           for key, envname, sub in CACHE_VARS},
        "data_root": cfg.cache.data_root or "./",
        "sizes": [s for s, _, _ in parse_repos(cfg.model.repos)] or [cfg.model.size],
        "modalities": [m for _, m, _ in parse_repos(cfg.model.repos)] or [cfg.model.modality],
        "domains": [d for _, _, d in parse_repos(cfg.model.repos)] or [cfg.model.domain],
        "active": cfg.model.active,
        "mode": cfg.gpu.mode or "slurm",
        "partition": cfg.gpu.partition or "",
        "time": cfg.gpu.time or "1:00:00",
        "mem": cfg.gpu.mem or "64G",
        "extra_args": cfg.gpu.extra_args or "",
        "retriever": cfg.retrieval.retriever or "qwen",
        "texlive_bin": cfg.paperprep.texlive_bin or "",
    }
    for k in ("sizes", "modalities", "domains"):
        st[k] = list(dict.fromkeys(st[k]))

    def _dim(msg, options, selected):
        choices = [Choice(o, checked=(o in selected)) for o in options]
        r = questionary.checkbox(msg, choices=choices, style=style,
                                 instruction="↑/↓ move · SPACE to toggle · ENTER to confirm").ask()
        if r is None:
            return list(selected)
        return list(dict.fromkeys(r)) or list(selected)

    # ---- steps. Each returns None(ok) | BACK | CANCEL -------------------
    def step_caches():
        # FIRST step. The router has no continue/back; each pane carries its
        # own continue (advances the form) + back (returns to the router).
        def _envvars_pane():
            # Reflect the LIVE environment, never a pre-filled guess: a var that
            # is already exported is shown read-only (✓ + path) -- the user's
            # environment owns it. An unset var (✗) is selectable; setting it
            # exports it into this process AND mkdir -p's the path so downstream
            # tools find it. Unset-and-left-blank stays empty (HF default cache).
            while True:
                rows = []
                for key, envname, _sub in CACHE_VARS:
                    cur = os.environ.get(envname)
                    if cur:
                        rows.append(Choice(f"✓ {envname:<16} {cur}",
                                           disabled="set in your environment"))
                    else:
                        rows.append(Choice(f"✗ {envname:<16} (not set — select to set)",
                                           value=envname))
                rows += [SEP(), CONTINUE, BACKROW]
                sc = _menu("Use env vars — ✓ already set (read-only) · ✗ select to set", rows)
                if sc == CANCEL:
                    return CANCEL
                if sc == BACKROW:
                    return BACK
                if sc == CONTINUE:
                    # Commit the live env values; vars still unset stay blank.
                    for key, envname, _sub in CACHE_VARS:
                        st[key] = os.environ.get(envname, "")
                    return None
                # sc is an unset var's envname (✓ rows are disabled / unselectable)
                for key, envname, sub in CACHE_VARS:
                    if sc == envname:
                        suggest = getattr(cfg.cache, key, "") or str(Path(_default_parent()) / sub)
                        v = _text(f"  set {envname}", suggest)
                        if v == CANCEL:
                            return CANCEL
                        v = v.strip()
                        if v:
                            path = str(Path(v).expanduser())
                            Path(path).mkdir(parents=True, exist_ok=True)
                            os.environ[envname] = path
                            st[key] = path
                        break

        def _parent_pane():
            while True:
                rows = [f"parent dir   [{st['cache_parent']}]", SEP(), CONTINUE, BACKROW]
                sc = _menu("Set parent dir — caches go in <parent>/{uv,hf,vllm,triton}", rows)
                if sc == CANCEL:
                    return CANCEL
                if sc == BACKROW:
                    return BACK
                if sc == CONTINUE:
                    # Env vars are the unified mechanism every downstream tool
                    # relies on. The parent-dir route therefore derives
                    # <parent>/{uv,hf,vllm,triton}, dynamically creates each
                    # (mkdir -p), AND exports it into the environment -- even when
                    # the user accepted the default without editing (otherwise a
                    # pre-set $HF_HOME like ~/.cache/hf silently wins from the seed
                    # above). Creating eagerly also surfaces write-permission
                    # problems here rather than mid-download.
                    for key, envname, sub in CACHE_VARS:
                        p = str((Path(st["cache_parent"]) / sub).expanduser())
                        Path(p).mkdir(parents=True, exist_ok=True)
                        os.environ[envname] = p
                        st[key] = p
                    return None
                p = _text("  parent dir", st["cache_parent"])
                if p == CANCEL:
                    return CANCEL
                st["cache_parent"] = p

        while True:
            choice = _menu("Caches — set/reuse env vars", ["Use env vars", "OR set parent dir"])
            if choice == CANCEL:
                return CANCEL
            r = _envvars_pane() if choice == "Use env vars" else _parent_pane()
            if r == CANCEL:
                return CANCEL
            if r is None:        # the pane's "continue ›" advances the form
                return None
            # r == BACK -> back to the router; re-show the two choices

    def step_data():
        while True:
            rows = [f"PAPERLENS_DATA_ROOT   {st['data_root']}", SEP(), CONTINUE, BACKROW]
            choice = _menu("Data root — reconstructed datasets + page images", rows)
            if choice == CANCEL:
                return CANCEL
            if choice == BACKROW:
                return BACK
            if choice == CONTINUE:
                return None
            v = _text("  PAPERLENS_DATA_ROOT", st["data_root"])
            if v == CANCEL:
                return CANCEL
            st["data_root"] = v

    def step_models():
        while True:
            rows = [
                f"size(s)        {', '.join(st['sizes']) or '(none)'}",
                f"modality(ies)  {', '.join(st['modalities']) or '(none)'}",
                f"domain(s)      {', '.join(st['domains']) or '(none)'}",
                SEP(), CONTINUE, BACKROW,
            ]
            choice = _menu("Models — edit a dimension, then continue", rows)
            if choice == CANCEL:
                return CANCEL
            if choice == BACKROW:
                return BACK
            if choice == CONTINUE:
                if st["sizes"] and st["modalities"] and st["domains"]:
                    return None
                continue
            if choice.startswith("size"):
                st["sizes"] = _dim("Model size(s)", list(SIZES), st["sizes"])
            elif choice.startswith("modality"):
                st["modalities"] = _dim("Modality(ies)", list(MODALITIES), st["modalities"])
            elif choice.startswith("domain"):
                st["domains"] = _dim("Domain(s)", list(DOMAINS), st["domains"])

    def step_active():
        repos = current_repos(st)
        if len(repos) == 1:
            st["active"] = repos[0]
            return None
        while True:
            rows = list(repos) + [SEP(), CONTINUE, BACKROW]
            r = _menu("Active model — pick one, then continue", rows,
                      default=st["active"] if st["active"] in repos else repos[0])
            if r == CANCEL:
                return CANCEL
            if r == BACKROW:
                return BACK
            if r == CONTINUE:
                if st["active"] not in repos:
                    st["active"] = repos[0]
                return None
            if r in repos:
                st["active"] = r

    def step_mode():
        # Branch selector. Picking 'slurm' advances straight to the slurm
        # options; 'local' first verifies a visible GPU. `‹ back` returns to the
        # previous step so the form stays uniformly navigable.
        while True:
            rows = ["slurm", "local", SEP(), BACKROW]
            r = _menu("GPU mode", rows, default=st["mode"])
            if r == CANCEL:
                return CANCEL
            if r == BACKROW:
                return BACK
            if r == "slurm":
                st["mode"] = "slurm"
                return None
            if r == "local":
                if _has_local_gpu():
                    st["mode"] = "local"
                    return None
                questionary.print(
                    "  ✗ no GPU visible here (nvidia-smi) — pick slurm to submit a job, "
                    "or re-run setup on a GPU node", style="fg:#cf222e")
                # loop: re-show the GPU mode choice

    def step_slurm():
        if st["mode"] != "slurm":
            return None
        while True:
            part_display = st["partition"] or "(REQUIRED — set this for your cluster)"
            rows = [
                f"partition     {part_display}",
                f"time          {st['time']}",
                f"host RAM      {st['mem']}",
                f"extra srun    {st['extra_args'] or '(none — e.g. --account=… --qos=… --constraint=…)'}",
                SEP(), CONTINUE, BACKROW,
            ]
            choice = _menu("Slurm settings — edit a field, then continue", rows)
            if choice == CANCEL:
                return CANCEL
            if choice == BACKROW:
                return BACK
            if choice == CONTINUE:
                if not st["partition"].strip():
                    questionary.print(
                        "  ✗ partition is required — your cluster's GPU partition (e.g. "
                        "gpu, ailab). Run `paperscorer setup --princeton` to fill the "
                        "Princeton defaults.", style="fg:#cf222e")
                    continue
                return None
            if choice.startswith("partition"):
                v = _text("  partition", st["partition"])
                if v == CANCEL: return CANCEL
                st["partition"] = v
            elif choice.startswith("time"):
                v = _text("  time limit", st["time"])
                if v == CANCEL: return CANCEL
                st["time"] = v
            elif choice.startswith("host RAM"):
                v = _text("  host RAM", st["mem"])
                if v == CANCEL: return CANCEL
                st["mem"] = v
            elif choice.startswith("extra srun"):
                v = _text("  extra srun args", st["extra_args"])
                if v == CANCEL: return CANCEL
                st["extra_args"] = v

    def step_texlive():
        # pdflatex is NOT guaranteed on every machine, and LaTeX-dir reviews
        # (paperprep's compile stage) hard-fail without it. Three paths:
        # (1) inherit PATH if pdflatex is already there, (2) override with a
        # custom bin dir, (3) auto-install TeX Live into a user-local dir
        # (writes a self-contained shell script + optionally runs it).
        from . import texlive as tl
        INSTALL = "install texlive"
        while True:
            cur = st["texlive_bin"]
            # Real probe (runs `pdflatex --version`) — `which` alone misses
            # broken installs / missing shared libs. Override is tried first
            # so the user sees their explicit choice reflected in the status.
            ok, path_pl = tl.pdflatex_works(bin_dir=cur or None)
            if cur and ok:
                status = f"✓ override works: {path_pl}"
            elif cur and not ok:
                status = f"✗ override set but pdflatex --version FAILS at {cur} — fix the path or reinstall"
            elif ok:
                status = f"✓ inheriting PATH: {path_pl}"
            else:
                status = "✗ pdflatex NOT runnable — LaTeX-dir reviews will fail without an override or install"
            rows = [
                Choice(f"status         {status}", disabled="informational"),
                f"override       {cur or '(empty — rely on PATH)'}",
                f"{INSTALL}  write+run setup_texlive.sh (scheme-medium, ~1GB, ~15-30 min)",
                SEP(), CONTINUE, BACKROW,
            ]
            choice = _menu("Texlive — pdflatex/latexmk for LaTeX-dir reviews", rows)
            if choice == CANCEL:
                return CANCEL
            if choice == BACKROW:
                return BACK
            if choice == CONTINUE:
                return None
            if isinstance(choice, str) and choice.startswith("override"):
                v = _text("  texlive bin dir (e.g. /usr/local/texlive/2025/bin/x86_64-linux)",
                         st["texlive_bin"])
                if v == CANCEL:
                    return CANCEL
                st["texlive_bin"] = v.strip()
            elif isinstance(choice, str) and choice.startswith(INSTALL):
                # Sub-pane: pick install dir, show plan, write script, optionally run.
                default_dir = str(Path.home() / "texlive")
                inst_dir = _text(
                    "  install dir (TeX Live will be written here; needs ~1GB free)",
                    default_dir)
                if inst_dir == CANCEL:
                    return CANCEL
                inst_dir = inst_dir.strip() or default_dir
                inst_dir = str(Path(inst_dir).expanduser())
                sp = tl.write_install_script(ROOT, inst_dir)
                bin_dir = tl.bin_dir_for(inst_dir)
                questionary.print(f"  ✓ wrote install script: {sp}", style="fg:#1a7f37")
                questionary.print(
                    f"  install will: download install-tl-unx from CTAN -> "
                    f"perl install-tl --no-interaction (~15-30 min) -> "
                    f"pdflatex lands at {bin_dir}/pdflatex", style="fg:#767676")
                run_now = confirm_menu(
                    "run setup_texlive.sh now? (blocks the wizard for the install)",
                    yes="run now ›", no="skip — I'll run it later",
                    default_yes=False)
                if run_now:
                    rc = tl.run_install_script(sp)
                    if rc != 0:
                        questionary.print(
                            f"  ✗ install script failed (rc={rc}) — re-run "
                            f"`bash {sp}` manually to debug", style="fg:#cf222e")
                    else:
                        # Verify the install actually produced a runnable
                        # pdflatex — rc=0 from install-tl doesn't guarantee
                        # the binary works (corrupt download, partial scheme,
                        # platform mismatch).
                        works, where = tl.pdflatex_works(bin_dir=bin_dir)
                        if not works:
                            questionary.print(
                                f"  ✗ install finished but `{bin_dir}/pdflatex "
                                f"--version` does not run — check the log "
                                f"above and re-run `bash {sp}`",
                                style="fg:#cf222e")
                        else:
                            st["texlive_bin"] = str(bin_dir)
                            # In-process PATH injection so paperprep-serve
                            # and the next wizard loop see pdflatex without
                            # needing a fresh shell. (bashrc append below
                            # is for FUTURE shells.)
                            tl.inject_bin_into_path(bin_dir)
                            questionary.print(
                                f"  ✓ pdflatex runs at {where}; texlive_bin "
                                f"set + injected into this session's PATH",
                                style="fg:#1a7f37")
                            if confirm_menu("append `export PATH=...` to ~/.bashrc? (for future shells)",
                                            yes="append ›", no="skip",
                                            default_yes=True):
                                added = tl.append_path_to_bashrc(bin_dir)
                                msg = "appended to ~/.bashrc" if added else "already in ~/.bashrc"
                                questionary.print(f"  ✓ {msg}", style="fg:#1a7f37")
                else:
                    # Pre-fill texlive_bin so future runs find pdflatex once the
                    # user runs the script themselves; no harm if they bail.
                    st["texlive_bin"] = str(bin_dir)
                    questionary.print(
                        f"  → run later: bash {sp}\n"
                        f"  → then add to PATH: export PATH={bin_dir}:$PATH\n"
                        f"  (texlive_bin pre-set to {bin_dir} for after the install)",
                        style="fg:#767676")

    def step_retriever():
        # Only present with --with-retrieval. Single-select (like step_mode):
        # arrow to a retriever to select it, then continue.
        opts = {
            "qwen": "qwen  (dense; FAISS index + Qwen3-Embedding-0.6B)",
            "bm25": "bm25  (lucene; index only, no model)",
        }
        while True:
            cur = opts.get(st["retriever"])
            rows = [opts["qwen"], opts["bm25"], SEP(), CONTINUE, BACKROW]
            choice = _menu("Retriever — pick one, then continue", rows, default=cur)
            if choice == CANCEL:
                return CANCEL
            if choice == BACKROW:
                return BACK
            if choice == CONTINUE:
                return None
            if choice == opts["qwen"]:
                st["retriever"] = "qwen"
            elif choice == opts["bm25"]:
                st["retriever"] = "bm25"

    steps = [step_caches, step_data, step_models, step_active, step_mode, step_slurm, step_texlive]
    if with_retrieval:
        steps.append(step_retriever)

    i = 0
    while i < len(steps):
        res = steps[i]()
        if res == CANCEL:
            return CANCEL
        if res == BACK:
            i = max(0, i - 1)
            while i > 0 and _is_noop(steps[i], st):
                i -= 1
            continue
        i += 1

    # ---- commit working state back to cfg -------------------------------
    repos = current_repos(st)
    cfg.cache.uv_cache = st["uv_cache"].strip()
    cfg.cache.hf_home = st["hf_home"].strip()
    cfg.cache.vllm_cache = st["vllm_cache"].strip()
    cfg.cache.triton_cache = st["triton_cache"].strip()
    cfg.cache.data_root = (st["data_root"] or "").strip()
    cfg.model.repos = repos
    cfg.model.active = st["active"] if st["active"] in repos else repos[0]
    cfg.gpu.mode = st["mode"]
    cfg.gpu.partition = st["partition"].strip()
    cfg.gpu.time = st["time"].strip()
    cfg.gpu.mem = st["mem"].strip()
    cfg.gpu.constraint = ""               # constraints live in extra_args now
    cfg.gpu.extra_args = (st["extra_args"] or "").strip()
    cfg.retrieval.retriever = st["retriever"]
    cfg.paperprep.texlive_bin = st["texlive_bin"].strip()
    return cfg


def current_repos(st: dict) -> list[str]:
    return [model_repo(s, m, d)
            for s in st["sizes"] for m in st["modalities"] for d in st["domains"]]


def parse_repos(repos: list[str]) -> list[tuple[str, str, str]]:
    from .config import parse_repo
    return [parse_repo(r) for r in (repos or [])]


def _is_noop(step, st: dict) -> bool:
    """Steps that auto-skip given current state — the back-walk must skip them
    or it bounces (e.g. back past slurm settings when mode=local)."""
    name = step.__name__
    if name == "step_slurm":
        return st["mode"] != "slurm"
    if name == "step_active":
        return len(current_repos(st)) <= 1
    return False
