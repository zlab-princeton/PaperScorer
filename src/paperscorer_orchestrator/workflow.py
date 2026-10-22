"""`paperscorer run <pdf>` — headless end-to-end review via the paperscorer-reviewing stack.

Generates a self-contained job script that:
  1. brings up paperprep-serve (its own venv) + paperlens-serve (pinned to the
     chosen downloaded model) + paperlensreview, all in the background,
  2. POSTs the PDF to the reviewing /submit API, polls /status,
  3. prints the verdict JSON and tears the stack down.

local gpu  -> the script runs directly (host must have a visible CUDA GPU).
slurm gpu  -> the whole script is wrapped in one `srun` allocation.

The training-inference venv (vLLM + llamafactory) is discovered like the
reviewing launcher does: $PAPERLENS_SERVE_VENV, else the canonical LF venv.
"""
from __future__ import annotations

import json
import shlex
import subprocess
import tempfile
from pathlib import Path

from rich.console import Console

from .config import Config, ROOT
from .gpu import srun_prefix
from .setup import REVIEWING, TRAININFER, PAPERPREP, ARXIV_SERVER

console = Console()

ARXIV_RETRIEVER = ARXIV_SERVER / "external" / "arxiv_retriever"


def _job_script(cfg: Config, pdf: Path, model_path: str, out_dir: Path) -> Path:
    template = "qwen2_vl" if cfg.model.modality == "vision" else "qwen"
    serve_yaml = out_dir / "serve.yaml"
    serve_yaml.write_text(f"""server: {{host: 127.0.0.1, port: 18402}}
model:
  ckpt_path: {model_path}
  template: {template}
  cutoff_len: 24480
  enable_thinking: false
  dtype: bfloat16
vllm: {{tensor_parallel_size: 1, pipeline_parallel_size: 1, gpu_memory_utilization: 0.65, max_model_len: 24580, disable_log_stats: true}}
scoring: {{positive_token: Accept, negative_token: Reject, decision_token_idx: 5, max_new_tokens: 8, image_max_pixels: 1003520, image_min_pixels: 784}}
logging: {{level: INFO}}
""")
    script = out_dir / "run_e2e.sh"
    review_venv = REVIEWING / ".venv"
    # vLLM venv for paperlens-serve: $PAPERLENS_SERVE_VENV else the
    # paperscorer-training-and-inference child venv that `paperscorer setup` provisions
    # (has vllm + paperlens_cli). No machine-specific hardcode.
    lf_venv = str(TRAININFER / ".venv")
    # Pin paperprep's pdflatex/latexmk lookup when the user configured a
    # texlive bin dir; otherwise let paperprep inherit $PATH (default).
    texlive_arg = (f"--texlive-bin {shlex.quote(cfg.paperprep.texlive_bin)} "
                   if cfg.paperprep.texlive_bin else "")
    script.write_text(f"""#!/usr/bin/env bash
set -euo pipefail
OUT={out_dir}
PP_PORT=18404; PL_PORT=18402; RV_PORT=18403; MINERU_PORT=40402
SERVE_VENV="${{PAPERLENS_SERVE_VENV:-{lf_venv}}}"
REVIEW_VENV="${{PAPERLENSREVIEW_VENV:-$SERVE_VENV}}"

echo "[run] node=$(hostname)"; nvidia-smi -L | head -1

# 1. paperprep serve + MinerU FIRST -- wait until it has reserved its GPU
#    memory before starting paperlens-serve. Parallel vLLM-engine startup
#    races for GPU memory and OOMs the second one.
env "PATH={PAPERPREP}/.venv/bin:$PATH" "{PAPERPREP}/.venv/bin/python" \\
    -m paper_anonymizer.paperprep.cli serve \\
    --output-dir "$OUT/paperprep_work" --host 127.0.0.1 --port $PP_PORT \\
    --mineru-port $MINERU_PORT --gpu-memory-utilization 0.20 \\
    {texlive_arg}> "$OUT/paperprep.log" 2>&1 &
PP=$!
echo "[run] waiting for paperprep/MinerU /healthz (up to 12 min)"
for _ in $(seq 1 144); do
  curl -sf http://127.0.0.1:$PP_PORT/healthz >/dev/null 2>&1 && break
  kill -0 $PP 2>/dev/null || {{ echo "[run] paperprep failed — see $OUT/paperprep.log" >&2; exit 1; }}
  sleep 5
done

# 2. paperscorer serve (pinned to the chosen model) -- now that MinerU's memory is reserved
env "PYTHONPATH={TRAININFER}/src" "$SERVE_VENV/bin/python" -m paperlens_cli serve \\
    --config "$OUT/serve.yaml" --port $PL_PORT > "$OUT/paperlens.log" 2>&1 &
PL=$!
echo "[run] waiting for paperscorer /health (up to 12 min)"
for _ in $(seq 1 144); do
  curl -sf http://127.0.0.1:$PL_PORT/health >/dev/null 2>&1 && break
  kill -0 $PL 2>/dev/null || {{ echo "[run] paperscorer failed — see $OUT/paperlens.log" >&2; exit 1; }}
  sleep 5
done

# 3. reviewing server
export PAPERPREP_SERVE_BASE_URL=http://127.0.0.1:$PP_PORT
export PAPERLENS_SERVE_BASE_URL=http://127.0.0.1:$PL_PORT
export PAPERLENS_REVIEW_MODALITY={cfg.model.modality}
export PAPERLENS_REVIEW_DOMAIN={cfg.model.domain}
env "PYTHONPATH={REVIEWING}/src" "$REVIEW_VENV/bin/python" -m paperlensreview serve \\
    --config {REVIEWING}/configs/server.yaml --port $RV_PORT --skip_preflight \\
    > "$OUT/paperlensreview.log" 2>&1 &
RV=$!
trap 'kill $RV $PL $PP 2>/dev/null; pkill -f mineru-vllm-server 2>/dev/null || true' EXIT
sleep 8

# 4. submit the PDF + poll
"$REVIEW_VENV/bin/python" - "$RV_PORT" "{pdf}" "$OUT/verdict.json" <<'PYEOF'
import sys, json, time, urllib.request
port, pdf, out = sys.argv[1], sys.argv[2], sys.argv[3]
base = f"http://127.0.0.1:{{port}}"
b = "----plrun"
from pathlib import Path
data = f'--{{b}}\\r\\nContent-Disposition: form-data; name="file"; filename="{{Path(pdf).name}}"\\r\\nContent-Type: application/pdf\\r\\n\\r\\n'.encode() + Path(pdf).read_bytes() + f'\\r\\n--{{b}}--\\r\\n'.encode()
req = urllib.request.Request(base+"/submit", data=data, method="POST")
req.add_header("Content-Type", f"multipart/form-data; boundary={{b}}")
jid = json.loads(urllib.request.urlopen(req, timeout=120).read())["job_id"]
print("[run] job", jid, flush=True)
for _ in range(200):
    s = json.loads(urllib.request.urlopen(base+f"/status/{{jid}}", timeout=10).read())
    if s["state"] in ("done","error"): break
    time.sleep(3)
Path(out).write_text(json.dumps(s, indent=2))
print("[run] state:", s["state"])
if s.get("result"): print("[run] verdict:", s["result"]["decision"], "p_accept=", round(s["result"]["p_accept"],4))
if s.get("error"): print("[run] error:", s["error"])
PYEOF
echo "[run] verdict written to $OUT/verdict.json"
""")
    script.chmod(0o755)
    return script


def preflight_serve(cfg: Config) -> bool:
    """Health check before serving/running: the active model is downloaded and
    the child venvs serve needs exist. Prints a checklist; True iff all green."""
    from .models import model_is_cached
    console.rule("[bold]Pre-flight")
    ok = True
    if model_is_cached(cfg):
        console.print(f"  [green]✓[/green] model cached  ({cfg.model.repo})")
    else:
        console.print(f"  [red]✗[/red] model not downloaded  ({cfg.model.repo}) — "
                      "run [cyan]paperscorer download[/cyan]")
        ok = False
    for name, p in (("paperprep", PAPERPREP), ("paperscorer-training-and-inference", TRAININFER),
                    ("paperscorer-reviewing", REVIEWING)):
        if (p / ".venv").exists():
            console.print(f"  [green]✓[/green] {name} venv")
        else:
            console.print(f"  [red]✗[/red] {name} venv missing ({p}/.venv) — "
                          "run [cyan]paperscorer setup[/cyan]")
            ok = False
    if not ok:
        console.print("\n  [red]pre-flight failed[/red] — fix the above, then retry.\n")
    return ok


def run_workflow(cfg: Config, pdf: Path, model_path: str) -> int:
    if not preflight_serve(cfg):
        return 2
    # Run dir MUST be on shared storage: in slurm mode the job script + its
    # verdict.json are written here on the login node but executed/read on the
    # compute node. /tmp is node-local, so anchor under the (scratch) repo root.
    runs = ROOT / ".paperlens_runs"
    runs.mkdir(exist_ok=True)
    out_dir = Path(tempfile.mkdtemp(prefix="run_", dir=runs))
    script = _job_script(cfg, pdf, model_path, out_dir)
    console.print(f"[dim]job dir: {out_dir}[/dim]")
    cmd = srun_prefix(cfg) + ["bash", str(script)]
    mode = "slurm" if cfg.gpu.mode == "slurm" else "local"
    console.print(f"[bold]running {mode} e2e[/bold] for {pdf.name} with {cfg.model.repo}")
    console.print(f"  {' '.join(cmd)}\n")
    rc = subprocess.run(cmd, env=cfg.export_env()).returncode
    verdict = out_dir / "verdict.json"
    if verdict.exists():
        v = json.loads(verdict.read_text())
        if v.get("result"):
            r = v["result"]
            console.rule("[bold]Verdict")
            console.print(f"  [bold]{r['decision']}[/bold]   p_accept = {r['p_accept']:.4f}   "
                          f"({cfg.model.size} {cfg.model.modality} / {cfg.model.domain})")
        else:
            console.print(f"[yellow]no verdict[/yellow]: state={v.get('state')} error={v.get('error')}")
    else:
        console.print("[red]no verdict.json produced[/red] — check logs in the job dir")
    return rc


def _serve_script(cfg: Config, model_path: str, out_dir: Path, port: int,
                  *, slurm: bool) -> Path:
    """Build the bash script the (s)run allocation executes.

    Slurm mode binds paperprep + paperlens-serve to 0.0.0.0 so the login node
    can reach them, emits programmatic URL markers (paperprep_url / paperlens_url)
    for the orchestrator to discover, and then just `wait`s — paperlensreview
    is launched on the login host because it needs internet for /submit_arxiv.

    Local mode binds to 127.0.0.1 and execs paperlensreview in the foreground
    (current single-host behavior; no orchestrator handoff needed).
    """
    template = "qwen2_vl" if cfg.model.modality == "vision" else "qwen"
    bind_host = "0.0.0.0" if slurm else "127.0.0.1"
    (out_dir / "serve.yaml").write_text(f"""server: {{host: {bind_host}, port: 18502}}
model:
  ckpt_path: {model_path}
  template: {template}
  cutoff_len: 24480
  enable_thinking: false
  dtype: bfloat16
vllm: {{tensor_parallel_size: 1, pipeline_parallel_size: 1, gpu_memory_utilization: 0.65, max_model_len: 24580, disable_log_stats: true}}
scoring: {{positive_token: Accept, negative_token: Reject, decision_token_idx: 5, max_new_tokens: 8, image_max_pixels: 1003520, image_min_pixels: 784}}
logging: {{level: INFO}}
""")
    # default scorer venv = the paperscorer-training-and-inference child venv (vllm + paperlens_cli);
    # override with $PAPERLENS_SERVE_VENV. No machine-specific hardcode.
    lf_venv = str(TRAININFER / ".venv")
    texlive_arg = (f"--texlive-bin {shlex.quote(cfg.paperprep.texlive_bin)} "
                   if cfg.paperprep.texlive_bin else "")

    if slurm:
        # Slurm: publish URLs and block on the two background daemons. The
        # login-side orchestrator launches paperlensreview separately.
        tail = """\
echo "@@PL paperprep_url http://$(hostname):$PP_PORT"
echo "@@PL paperlens_url http://$(hostname):$PL_PORT"
echo "[serve] compute-side daemons ready; reviewing UI runs on the login host"
# Keep the allocation alive until either daemon dies or we get a signal.
wait $PP $PL
"""
    else:
        # Local: launch paperlensreview in the foreground, same host (no internet
        # caveat). Reviewing finds paperprep+paperscorer via 127.0.0.1.
        tail = f"""\
export PAPERPREP_SERVE_BASE_URL=http://127.0.0.1:$PP_PORT
export PAPERLENS_SERVE_BASE_URL=http://127.0.0.1:$PL_PORT
echo "@@PL url PaperScorer reviewing UI:  http://$(hostname):$RV_PORT"
exec env "PYTHONPATH={REVIEWING}/src" PYTHONUNBUFFERED=1 "$SERVE_VENV/bin/python" -m paperlensreview serve \\
    --config {REVIEWING}/configs/server.yaml --port $RV_PORT --skip_preflight
"""

    script = out_dir / "serve_ui.sh"
    script.write_text(f"""#!/usr/bin/env bash
set -euo pipefail
OUT={out_dir}
PP_PORT=18504; PL_PORT=18502; RV_PORT={port}; MINERU_PORT=40502
SERVE_VENV="${{PAPERLENS_SERVE_VENV:-{lf_venv}}}"

echo "[serve] node=$(hostname)"; nvidia-smi -L | head -1
PP=""; PL=""
trap 'echo; echo "[serve] shutting down"; kill $PL $PP 2>/dev/null; pkill -f mineru-vllm-server 2>/dev/null || true' EXIT INT TERM

# Serialized startup: bring up paperprep+MinerU FIRST and wait until it has
# fully loaded (and reserved its GPU memory) before starting paperlens-serve.
# Launching both vLLM engines in parallel races for GPU memory and OOMs the
# second engine ("Available KV cache memory: -X GiB").
env "PATH={PAPERPREP}/.venv/bin:$PATH" PYTHONUNBUFFERED=1 "{PAPERPREP}/.venv/bin/python" \\
    -m paper_anonymizer.paperprep.cli serve \\
    --output-dir "$OUT/paperprep_work" --host {bind_host} --port $PP_PORT \\
    --mineru-port $MINERU_PORT --gpu-memory-utilization 0.20 \\
    {texlive_arg}> "$OUT/paperprep.log" 2>&1 &
PP=$!
pp_ready=0
for _ in $(seq 1 144); do
  if curl -sf http://127.0.0.1:$PP_PORT/healthz >/dev/null 2>&1; then pp_ready=1; echo "@@PL mineru ready"; break; fi
  if ! kill -0 $PP 2>/dev/null; then echo "@@PL mineru failed"; break; fi
  sleep 5
done
if [ $pp_ready -ne 1 ]; then
  echo "[serve] paperprep/MinerU failed to start — see $OUT/paperprep.log" >&2
  exit 1
fi

# MinerU's memory is now reserved; start paperlens-serve against the remaining budget.
env "PYTHONPATH={TRAININFER}/src" PYTHONUNBUFFERED=1 "$SERVE_VENV/bin/python" -m paperlens_cli serve \\
    --config "$OUT/serve.yaml" --port $PL_PORT > "$OUT/paperlens.log" 2>&1 &
PL=$!
pl_ready=0
for _ in $(seq 1 144); do
  if curl -sf http://127.0.0.1:$PL_PORT/health >/dev/null 2>&1; then pl_ready=1; echo "@@PL paperscorer ready"; break; fi
  if ! kill -0 $PL 2>/dev/null; then echo "@@PL paperscorer failed"; break; fi
  sleep 5
done
if [ $pl_ready -ne 1 ]; then
  echo "[serve] paperlens-serve failed to start — see $OUT/paperlens.log" >&2
  exit 1
fi

{tail}""")
    script.chmod(0o755)
    return script


def serve_ui(cfg: Config, model_path: str, *, port: int = 8003) -> int:
    """Bring up the reviewing UI stack with the chosen model and keep it running.

    Local mode  — paperlensreview runs inside the (s)run script on the same host.
    Slurm mode  — paperlensreview runs on the LOGIN host (this process); only
                  paperprep-serve + paperlens-serve live inside the slurm
                  allocation. The split is required for /submit_arxiv: the
                  reviewing API needs internet access, which compute nodes
                  on della don't have. paperlensreview reaches the compute
                  daemons via the URL markers the script publishes.

    Ctrl-C tears everything down cleanly:
      - the slurm allocation via the script's trap + an explicit `scancel`
      - the login-side paperlensreview process (slurm mode)
    """
    import os
    import signal as _signal
    import shutil as _shutil
    import socket
    from .progress import run_with_progress

    if not preflight_serve(cfg):
        return 2
    runs = ROOT / ".paperlens_runs"; runs.mkdir(exist_ok=True)
    out_dir = Path(tempfile.mkdtemp(prefix="serve_", dir=runs))
    slurm = cfg.gpu.mode == "slurm"
    script = _serve_script(cfg, model_path, out_dir, port, slurm=slurm)

    job_name = f"pl-serve-{out_dir.name.split('_')[-1]}"   # unique per invocation
    prefix = srun_prefix(cfg)
    if slurm:
        prefix = prefix[:1] + [f"--job-name={job_name}"] + prefix[1:]   # srun --job-name=… …
    cmd = prefix + ["bash", str(script)]

    console.print(f"[bold]serving UI ({'slurm' if slurm else 'local'})[/bold] with {cfg.model.repo}")
    console.print(f"  [dim]logs: {out_dir} · Ctrl-C to stop"
                  + (" (cancels the slurm job)" if slurm else "") + "[/dim]")
    if slurm:
        # Show the login-side URL upfront so the user can prepare the SSH tunnel
        # while the compute stack is still warming up.
        login_host = socket.gethostname()
        console.print(f"  [bold]reviewing URL (when ready):[/bold] http://{login_host}:{port}")
        console.print(f"  [dim]SSH tunnel:[/dim] ssh -N -L {port}:{login_host}:{port} <user>@{login_host}\n")
    else:
        console.print("")

    # Mutable boxes so the marker callback + interrupt handler share state.
    review_proc_box: dict = {"proc": None}
    urls: dict = {"paperprep_url": None, "paperlens_url": None}

    def _spawn_reviewing_on_login() -> None:
        """Launch paperlensreview on this host, pointed at the compute-side
        paperprep+paperscorer URLs. Idempotent: only spawns once."""
        if review_proc_box["proc"] is not None:
            return
        env = dict(cfg.export_env())
        env["PAPERPREP_SERVE_BASE_URL"] = urls["paperprep_url"]
        env["PAPERLENS_SERVE_BASE_URL"] = urls["paperlens_url"]
        # Match the reviewing server's sharegpt modality/domain to the loaded
        # model: a text model rejects the vision row's <image> tags otherwise.
        env["PAPERLENS_REVIEW_MODALITY"] = str(cfg.model.modality)
        env["PAPERLENS_REVIEW_DOMAIN"] = str(cfg.model.domain)
        # Force line-buffered stdout/stderr so log lines appear in the live
        # dashboard panel immediately, not after the 8KB block buffer fills.
        env["PYTHONUNBUFFERED"] = "1"
        review_python = REVIEWING / ".venv" / "bin" / "python"
        if not review_python.exists():
            # Fall back to the LF venv (provisioned by `paperscorer setup`).
            review_python = TRAININFER / ".venv" / "bin" / "python"
        env["PYTHONPATH"] = f"{REVIEWING / 'src'}:{env.get('PYTHONPATH', '')}".rstrip(":")
        argv = [
            str(review_python), "-m", "paperlensreview", "serve",
            "--config", str(REVIEWING / "configs" / "server.yaml"),
            "--port", str(port), "--skip_preflight",
        ]
        review_log = open(out_dir / "paperlensreview.log", "w")
        review_proc_box["proc"] = subprocess.Popen(
            argv, env=env, stdout=review_log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    def _on_marker(key: str, value: str) -> None:
        if not slurm:
            return
        if key in urls:
            urls[key] = value
            if all(urls.values()):
                _spawn_reviewing_on_login()

    def _interrupt(proc):
        # Tear down login-side reviewing first (so the user sees the slurm cancel
        # banner last), then signal the (s)run, then belt-and-suspenders scancel.
        rp = review_proc_box.get("proc")
        if rp is not None and rp.poll() is None:
            try:
                os.killpg(os.getpgid(rp.pid), _signal.SIGTERM)
            except ProcessLookupError:
                pass
        try:
            os.killpg(os.getpgid(proc.pid), _signal.SIGINT)
        except ProcessLookupError:
            pass
        if slurm and _shutil.which("scancel"):
            subprocess.run(["scancel", "--name", job_name], stderr=subprocess.DEVNULL)

    try:
        rc = run_with_progress(
            cmd, cfg.export_env(),
            services=[("mineru", "paperprep-serve (MinerU OCR)"),
                      ("paperscorer", "paperlens-serve (scorer)")],
            log_files=[("paperprep-serve", out_dir / "paperprep.log"),
                       ("paperlens-serve", out_dir / "paperlens.log")]
                      + ([("paperlensreview", out_dir / "paperlensreview.log")] if slurm else []),
            interrupt=_interrupt,
            on_marker=_on_marker,
        )
    finally:
        # If anything (success / crash / Ctrl-C) ends the loop, make sure the
        # login-side reviewing process doesn't leak.
        rp = review_proc_box.get("proc")
        if rp is not None and rp.poll() is None:
            try:
                os.killpg(os.getpgid(rp.pid), _signal.SIGTERM)
                rp.wait(timeout=10)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                pass
    return rc


# ---------------------------------------------------------------------------
# `paperscorer serve --with-retrieval` — search/rerank stack + UI
# ---------------------------------------------------------------------------

def _render_arxiv_server_config(cfg: Config, out_dir: Path, *, pl_port: int,
                                rt_port: int, db_path: Path, images_root) -> Path:
    """Concrete paperscorer-arxiv-server config wired to this serve's ports/paths."""
    p = out_dir / "arxiv_server.yaml"
    p.write_text(
        "# generated by `paperscorer serve --with-retrieval`\n"
        "server: {host: 0.0.0.0, port: 8000}\n"
        f"retriever: {{base_url: 'http://127.0.0.1:{rt_port}', topk_retrieve: 200, timeout_seconds: 30.0}}\n"
        f"paperlens_serve: {{base_url: 'http://127.0.0.1:{pl_port}', timeout_seconds: 600.0}}\n"
        f"reranker: {{domain: {cfg.model.domain}, modality: {cfg.model.modality}}}\n"
        f"images_root: {images_root or ''}\n"
        f"cache: {{db_path: '{db_path}'}}\n"
        "search: {default_blend: default, blend_weight_retriever: 0.3, "
        "blend_weight_p_accept: 0.7, rerank_pool: 50, compare_top: 20}\n"
    )
    return p


def _serve_retrieval_script(cfg: Config, model_path: str, out_dir: Path, port: int,
                            retr_cfg: Path, as_cfg: Path, pred_parquet, envs_path: str,
                            db_path: Path) -> Path:
    template = "qwen2_vl" if cfg.model.modality == "vision" else "qwen"
    (out_dir / "serve.yaml").write_text(f"""server: {{host: 127.0.0.1, port: 18602}}
model:
  ckpt_path: {model_path}
  template: {template}
  cutoff_len: 24480
  enable_thinking: false
  dtype: bfloat16
vllm: {{tensor_parallel_size: 1, pipeline_parallel_size: 1, gpu_memory_utilization: 0.45, max_model_len: 24580, disable_log_stats: true}}
scoring: {{positive_token: Accept, negative_token: Reject, decision_token_idx: 5, max_new_tokens: 8, image_max_pixels: 1003520, image_min_pixels: 784}}
logging: {{level: INFO}}
""")
    lf_venv = str(TRAININFER / ".venv")
    as_venv = str(ARXIV_SERVER / ".venv")
    retr_py = f"{envs_path}/retriever/bin/python" if envs_path else "python"
    retr_src = str(ARXIV_RETRIEVER / "src")
    retr_server = str(ARXIV_RETRIEVER / "src" / "arxiv_retriever" / "server" / "retrieval_server.py")
    bootstrap = str(ARXIV_SERVER / "scripts" / "bootstrap_cache_from_litsearch.py")
    pred = str(pred_parquet) if pred_parquet else ""
    script = out_dir / "serve_retrieval.sh"
    script.write_text(f"""#!/usr/bin/env bash
set -uo pipefail
OUT={out_dir}
PL_PORT=18602; RT_PORT=18601; AS_PORT={port}
SERVE_VENV="${{PAPERLENS_SERVE_VENV:-{lf_venv}}}"
AS_VENV="{as_venv}"
RETR_PY="{retr_py}"
PRED="{pred}"
DB="{db_path}"

echo "[serve] node=$(hostname)"; nvidia-smi -L | head -1

# 1. reranker scorer (vLLM)
env "PYTHONPATH={TRAININFER}/src" "$SERVE_VENV/bin/python" -m paperlens_cli serve \\
    --config "$OUT/serve.yaml" --port $PL_PORT > "$OUT/paperlens.log" 2>&1 &
PL=$!

# 2. arxiv_retriever (embed model + index) in the `retriever` conda env
env "PYTHONPATH={retr_src}" "$RETR_PY" "{retr_server}" \\
    --config "{retr_cfg}" "server.port=$RT_PORT" > "$OUT/retriever.log" 2>&1 &
RT=$!
trap 'echo; echo "[serve] shutting down"; kill $PL $RT 2>/dev/null || true' EXIT INT TERM

# Independent readiness (scorer /health, retriever /state) with failure detection.
pl_ready=0; rt_ready=0
for _ in $(seq 1 180); do
  if [ $pl_ready -eq 0 ]; then
    if curl -sf http://127.0.0.1:$PL_PORT/health >/dev/null 2>&1; then pl_ready=1; echo "@@PL scorer ready"; fi
    if ! kill -0 $PL 2>/dev/null; then echo "@@PL scorer failed"; break; fi
  fi
  if [ $rt_ready -eq 0 ]; then
    if curl -sf http://127.0.0.1:$RT_PORT/state >/dev/null 2>&1; then rt_ready=1; echo "@@PL retriever ready"; fi
    if ! kill -0 $RT 2>/dev/null; then echo "@@PL retriever failed"; break; fi
  fi
  [ $pl_ready -eq 1 ] && [ $rt_ready -eq 1 ] && break
  sleep 5
done
if [ $pl_ready -ne 1 ] || [ $rt_ready -ne 1 ]; then
  echo "[serve] a service failed to start — see $OUT/{{paperscorer,retriever}}.log" >&2
  exit 1
fi

# 3. seed the p_accept cache from the predictions parquet, keyed by the scorer's
#    actual ckpt_path (so the running server's cache lookups hit these rows).
if [ -n "$PRED" ]; then
  CKPT=$("$AS_VENV/bin/python" -c "import json,urllib.request;print(json.load(urllib.request.urlopen('http://127.0.0.1:$PL_PORT/health')).get('ckpt_path',''))" 2>/dev/null || true)
  if [ -n "$CKPT" ]; then
    "$AS_VENV/bin/python" "{bootstrap}" --parquet "$PRED" --ckpt_path "$CKPT" --db_path "$DB" \\
        > "$OUT/cache_bootstrap.log" 2>&1 && echo "@@PL cache ready" || echo "@@PL cache failed"
  else
    echo "[serve] could not read scorer ckpt_path; cache starts empty" >&2; echo "@@PL cache failed"
  fi
else
  echo "[serve] no predictions parquet found; cache starts empty" >&2; echo "@@PL cache ready"
fi

# 4. arxiv-server (search UI + /search_compare) in the FOREGROUND
echo "@@PL url PaperScorer search UI:  http://$(hostname):$AS_PORT"
echo "@@PL url SSH tunnel:  ssh -N -L $AS_PORT:$(hostname):$AS_PORT <user>@<login-host>"
exec env "PYTHONPATH={ARXIV_SERVER}/src" "$AS_VENV/bin/python" -m paperlens_arxiv_server.server \\
    --config "{as_cfg}" --port $AS_PORT
""")
    script.chmod(0o755)
    return script


def serve_retrieval(cfg: Config, model_path: str, *, port: int = 8000) -> int:
    """`paperscorer serve --with-retrieval`: reranker scorer + arxiv_retriever
    (embed + index) + the paperscorer-arxiv-server search UI, with the p_accept
    cache bootstrapped from the HF predictions parquet. Ctrl-C tears it down.
    """
    import os
    import signal as _signal
    import shutil as _shutil
    from . import retrieval as R
    from .progress import run_with_progress

    # resolve the chosen retriever's index + (qwen) embed model
    which = cfg.retrieval.retriever if cfg.retrieval.retriever in R.RETRIEVERS else "qwen"
    paths, _src = R.resolve_index_paths(cfg, which)
    if not paths:
        console.print("[red]retrieval index not found[/red] — run "
                      "[cyan]paperscorer setup --with-retrieval[/cyan] first.")
        return 2
    embed = R.resolve_embed_model(cfg) if which == "qwen" else None
    if which == "qwen" and not embed:
        console.print("[red]Qwen3-Embedding-0.6B not cached[/red] — run "
                      "[cyan]paperscorer setup --with-retrieval[/cyan].")
        return 2
    if not (ARXIV_SERVER / ".venv").exists():
        console.print(f"[red]arxiv-server venv missing[/red] ({ARXIV_SERVER}/.venv) — "
                      "run [cyan]paperscorer setup --with-retrieval[/cyan].")
        return 2
    envs_path = R.resolve_conda_envs_path(cfg, dry_run=True)
    if not R.conda_env_exists(envs_path):
        console.print("[red]`retriever` conda env missing[/red] — run "
                      "[cyan]paperscorer setup --with-retrieval[/cyan].")
        return 2

    runs = ROOT / ".paperlens_runs"; runs.mkdir(exist_ok=True)
    out_dir = Path(tempfile.mkdtemp(prefix="serve_retr_", dir=runs))
    retr_cfg = R.render_retriever_config(which, paths, embed,
                                         out_path=out_dir / "retriever.yaml",
                                         port=18601)
    db_path = ARXIV_SERVER / "cache" / "p_accept.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    as_cfg = _render_arxiv_server_config(
        cfg, out_dir, pl_port=18602, rt_port=18601, db_path=db_path,
        images_root=R.resolve_images_root(cfg))
    pred = R.resolve_predictions_parquet(cfg)
    script = _serve_retrieval_script(cfg, model_path, out_dir, port, retr_cfg,
                                     as_cfg, pred, envs_path, db_path)

    slurm = cfg.gpu.mode == "slurm"
    job_name = f"pl-retr-{out_dir.name.split('_')[-1]}"
    prefix = srun_prefix(cfg)
    if slurm:
        prefix = prefix[:1] + [f"--job-name={job_name}"] + prefix[1:]
    cmd = prefix + ["bash", str(script)]

    console.print(f"[bold]serving retrieval UI ({'slurm' if slurm else 'local'})[/bold] "
                  f"· retriever={which} · reranker={cfg.model.repo}")
    console.print(f"  [dim]logs: {out_dir} · Ctrl-C to stop"
                  + (" (cancels the slurm job)" if slurm else "") + "[/dim]\n")

    def _interrupt(proc):
        try:
            os.killpg(os.getpgid(proc.pid), _signal.SIGINT)
        except ProcessLookupError:
            pass
        if slurm and _shutil.which("scancel"):
            subprocess.run(["scancel", "--name", job_name], stderr=subprocess.DEVNULL)

    return run_with_progress(
        cmd, cfg.export_env(),
        services=[("scorer", "paperlens-serve (reranker)"),
                  ("retriever", f"arxiv_retriever ({which} embed+index)"),
                  ("cache", "p_accept cache bootstrap")],
        log_files=[("paperlens-serve", out_dir / "paperlens.log"),
                   ("arxiv_retriever", out_dir / "retriever.log"),
                   ("cache bootstrap", out_dir / "cache_bootstrap.log")],
        interrupt=_interrupt,
        file_lines=7, main_lines=6,   # 3 log boxes + main: keep it inside a normal terminal
    )
