"""Smoke test: build two agent workspaces from a real paperprep output dir
and run claude on one + codex on the other in parallel, streaming events
to stdout so we can confirm the end-to-end pipeline before wiring the UI.

Run with the paperlens-reviewing venv's python:
    /scratch/gpfs/ZHUANGL/sk7524/PaperLens/tools/paperlens-reviewing/.venv/bin/python \
        scripts/agent_smoke.py
"""
from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from paperlensreview import agents


PAPERPREP_OUT = Path(
    "/scratch/gpfs/ZHUANGL/sk7524/PaperLens/.paperlens_runs/serve_5qcbv0ns/paperprep_work/258578a99b47_376ca88e"
)
SMOKE_ROOT = Path("/tmp/paperlens_agent_smoke")
SID = "smoketest"
TITLE = "smoke test paper"


def _find_text_md(pp: Path) -> Path | None:
    for cand in pp.rglob("normalized/**/*.md"):
        return cand
    for cand in pp.rglob("*.md"):
        return cand
    return None


def main() -> int:
    probe = agents.probe_agents()
    print(f"probe: {json.dumps(probe, indent=2)}")
    for k in ("claude", "codex"):
        if not probe.get(k, {}).get("available"):
            print(f"!! {k} not available; aborting", file=sys.stderr)
            return 2

    # Build the two workspaces (separate pairs per agent so they don't collide).
    SMOKE_ROOT.mkdir(parents=True, exist_ok=True)
    text_md = _find_text_md(PAPERPREP_OUT)
    panel = PAPERPREP_OUT / "panel.png"
    panel_path = panel if panel.exists() else None

    # Two runs to cover all four code paths: claude does no_prior, codex does
    # with_prior. Each tests its CLI's stream parser AND a different prompt.
    pairs: list[tuple[str, str, dict]] = []
    for agent_name, variant in (("claude", "no_prior"), ("codex", "with_prior")):
        ws = SMOKE_ROOT / agent_name / variant
        manifest = agents.build_workspace(
            dest=ws,
            paperprep_output_dir=PAPERPREP_OUT,
            submission_id=SID,
            modality="vision",
            panel_path=panel_path,
            text_md_path=text_md,
            paper_title=TITLE,
            with_prior=(variant == "with_prior"),
            prior_decision="Reject",
            prior_p_accept=0.31,
            agent=agent_name,
        )
        pairs.append((agent_name, variant, manifest))
        print(f"built {agent_name}/{variant}: workspace={ws} pages={manifest['n_images']}")

    # Quick sanity check on workspace layout.
    sample_ws = pairs[0][2]["workspace"]
    print(f"\nworkspace tree (sample {sample_ws}):")
    for p in sorted(Path(sample_ws).rglob("*"))[:25]:
        print(" ", p.relative_to(sample_ws))

    # Spawn one thread per (agent, variant) pair (just 2 here).
    started = time.time()
    results: dict[tuple[str, str], agents.AgentResult] = {}
    locks = threading.Lock()

    def _runner(agent_name: str, variant: str, manifest: dict) -> None:
        tag = f"{agent_name}/{variant}"
        last_kind = ""
        def _on_event(ev: dict) -> None:
            nonlocal last_kind
            kind = ev.get("kind", "?")
            p = ev.get("payload", {})
            if kind == "text":
                snippet = (p.get("text") or "").splitlines()[0][:120]
                print(f"[{tag}] text: {snippet}", flush=True)
            elif kind == "tool_call":
                print(f"[{tag}] tool: {p.get('name')} {json.dumps(p.get('input', {}))[:140]}", flush=True)
            elif kind == "tool_result":
                content = (p.get("content") or "")[:140].replace("\n", " ")
                print(f"[{tag}] result: {content}", flush=True)
            elif kind == "status":
                print(f"[{tag}] status: {p}", flush=True)
            elif kind == "error":
                print(f"[{tag}] ERROR: {p}", flush=True)
            elif kind == "done":
                preds = p.get("predictions")
                print(f"[{tag}] DONE elapsed={p.get('elapsed_s')}s decision={preds and preds.get('decision')}", flush=True)
            last_kind = kind

        try:
            res = agents.run_agent_headless(
                agent=agent_name,
                workspace=Path(manifest["workspace"]),
                prompt=manifest["prompt"],
                submission_id=SID,
                on_event=_on_event,
                timeout_s=900.0,
            )
        except Exception as e:
            print(f"[{tag}] runner crashed: {e}", flush=True)
            return
        with locks:
            results[(agent_name, variant)] = res

    threads = []
    for an, v, m in pairs:
        t = threading.Thread(target=_runner, args=(an, v, m),
                             daemon=True, name=f"smoke-{an}-{v}")
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    elapsed = time.time() - started
    n = len(pairs)
    print(f"\n=== summary  ({elapsed:.1f}s total) ===")
    ok = 0
    for (an, v), res in results.items():
        preds = res.predictions or {}
        print(f"{an}/{v:11s}  ok={res.ok}  exit={res.exit_code}  "
              f"decision={preds.get('decision')!r:10s} "
              f"rating={preds.get('rating')!r}  workspace={res.workspace}")
        if res.ok and res.predictions:
            ok += 1
        if res.error:
            print(f"  error: {res.error[:200]}")
    print(f"\n{ok}/{n} succeeded")
    return 0 if ok == n else 1


if __name__ == "__main__":
    sys.exit(main())
