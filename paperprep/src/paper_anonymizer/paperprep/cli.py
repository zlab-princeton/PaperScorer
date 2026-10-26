"""paperprep CLI — `paperprep run --input-manifest ...`.

End-to-end driver: orchestrates compile -> page-count filter -> mineru ->
normalize -> arxiv filters -> ShareGPT export. Each stage either runs
locally (`--workers N`) or via submitit (`--slurm`) with per-stage
GPU/CPU partitions.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

from paper_anonymizer.paperprep import manifest as mfst
from paper_anonymizer.paperprep import submit as sub
from paper_anonymizer.paperprep.progress import submitit_progress
from paper_anonymizer.paperprep.state import StateLog, StageResult
from paper_anonymizer.paperprep.stages import (
    compile as stage_compile,
    detect as stage_detect,
    export as stage_export,
    filter as stage_filter,
    mineru as stage_mineru,
    normalize as stage_normalize,
)


ALL_STAGES = ["compile", "mineru", "normalize", "filter", "export"]


def parse_overrides(spec: str | None) -> dict[str, dict]:
    """Parse `--stage-overrides 'mineru.timeout_min=480,compile.mem_gb=16'`.

    Returns {stage: {key: value}}. Values coerced to int when possible.
    """
    if not spec:
        return {}
    out: dict[str, dict] = {}
    for entry in spec.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if "=" not in entry or "." not in entry:
            raise ValueError(f"--stage-overrides format: stage.key=value (got {entry!r})")
        lhs, rhs = entry.split("=", 1)
        stage, key = lhs.split(".", 1)
        try:
            val: object = int(rhs)
        except ValueError:
            try:
                val = float(rhs)
            except ValueError:
                val = rhs
        out.setdefault(stage.strip(), {})[key.strip()] = val
    return out


def _run_local(stage_name: str, items: list, fn, *, chunk_size: int) -> list:
    """Local fallback when --slurm is not set. Runs serially per chunk."""
    chunks = sub.chunk(items, chunk_size)
    results: list = []
    for c in chunks:
        results.extend(fn(c))
    return results


def _flatten_submitit_results(jobs: list) -> list:
    """Each submitit job returns a list (one StageResult per paper). Flatten."""
    flat: list = []
    for j in jobs:
        try:
            res = j.result()
        except Exception as e:
            # The whole chunk failed; we can't attribute to specific papers
            print(f"WARNING: submitit job {j.job_id} failed: {e}", file=sys.stderr)
            continue
        flat.extend(res or [])
    return flat


def _dispatch(
    stage_name: str,
    items: list,
    fn,
    *,
    use_slurm: bool,
    cpu_partition: str,
    gpu_partition: str,
    chunk_size: int,
    logs_dir: Path,
    overrides: dict,
) -> list:
    if not items:
        return []
    if use_slurm:
        jobs = sub.submit_stage(
            stage_name=stage_name,
            items=items,
            fn=fn,
            cpu_partition=cpu_partition,
            gpu_partition=gpu_partition,
            chunk_size=chunk_size,
            logs_dir=logs_dir,
            overrides=overrides.get(stage_name),
        )
        submitit_progress(jobs, stage_name)
        return _flatten_submitit_results(jobs)
    return _run_local(stage_name, items, fn, chunk_size=chunk_size)


def cmd_run(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    logs_dir = (Path(args.logs_dir or output_dir / "logs") / run_id).resolve()
    logs_dir.mkdir(parents=True, exist_ok=True)

    state = StateLog(output_dir / "state.jsonl")
    overrides = parse_overrides(args.stage_overrides)
    requested = args.stages.split(",") if args.stages else ALL_STAGES
    bad = [s for s in requested if s not in ALL_STAGES]
    if bad:
        print(f"ERROR: unknown stages: {bad}; known: {ALL_STAGES}", file=sys.stderr)
        return 2

    # Load manifest
    if args.input_manifest:
        entries = mfst.load_manifest(Path(args.input_manifest))
    elif args.input_dir:
        entries = mfst.build_manifest_from_dir(Path(args.input_dir), args.input_mode)
    else:
        print("ERROR: --input-manifest OR (--input-dir + --input-mode) required", file=sys.stderr)
        return 2

    print(f"[paperprep] run_id={run_id}, {len(entries)} papers, output_dir={output_dir}")
    print(f"[paperprep] stages: {requested}, slurm={args.slurm}, "
          f"chunk_size={args.chunk_size}, logs_dir={logs_dir}")

    # ===== compile =====
    pdf_paths: list[tuple[str, Path]] = []
    if "compile" in requested:
        # Skip resume-OK ids by partial filtering (we still need them as inputs to next stage)
        to_run = entries if not args.resume else [
            e for e in entries if not state.already_ok(e.id, "compile")
        ]
        if to_run:
            results = _dispatch(
                "compile", to_run,
                lambda chunk: stage_compile.run_chunk(
                    chunk, output_dir, timeout=args.compile_timeout, texlive_bin=args.texlive_bin),
                use_slurm=args.slurm, cpu_partition=args.cpu_partition,
                gpu_partition=args.gpu_partition, chunk_size=args.chunk_size,
                logs_dir=logs_dir, overrides=overrides,
            )
            state.append_many(results)
        # Collect pdf_paths from latest state
        for e in entries:
            if state.already_ok(e.id, "compile"):
                # Find the output path from the latest log row
                pdf = _lookup_output_path(state, e.id, "compile") or (
                    output_dir / "compiled" / f"{e.id}.pdf"
                )
                pdf_paths.append((e.id, Path(pdf)))
            elif e.type == "pdf":
                # type=pdf entries are valid even though compile was skipped
                pdf_paths.append((e.id, e.path))
    else:
        # No compile stage requested; pass through input PDFs only
        for e in entries:
            if e.type == "pdf":
                pdf_paths.append((e.id, e.path))

    # ===== inline page-count filter (drop pdf < 4 pages) =====
    kept, dropped = stage_detect.filter_short_pdfs(pdf_paths)
    state.append_many(dropped)
    if not kept:
        print("[paperprep] no papers survived page-count filter")
        _print_summary(state)
        return 0
    print(f"[paperprep] page-count filter: kept {len(kept)} / {len(pdf_paths)}")

    # ===== mineru =====
    mineru_outputs: dict[str, dict] = {}
    if "mineru" in requested:
        to_run = kept if not args.resume else [
            (i, p) for i, p in kept if not state.already_ok(i, "mineru")
        ]
        results = _dispatch(
            "mineru", to_run,
            lambda chunk: stage_mineru.run_chunk(chunk, output_dir, max_pages=args.max_pages),
            use_slurm=args.slurm, cpu_partition=args.cpu_partition,
            gpu_partition=args.gpu_partition, chunk_size=min(args.chunk_size, 20),
            logs_dir=logs_dir, overrides=overrides,
        )
        state.append_many(results)
    # Collect mineru outputs from state for everything that's now ok
    for paper_id, pdf_path in kept:
        if state.already_ok(paper_id, "mineru"):
            mineru_dir = output_dir / "mineru" / paper_id
            cl = _find_first(mineru_dir, "*_content_list.json")
            mj = _find_first(mineru_dir, "*_middle.json")
            if cl is not None:
                mineru_outputs[paper_id] = {
                    "pdf_path": pdf_path,
                    "content_list_path": cl,
                    "middle_json_path": mj,
                }

    # ===== normalize =====
    normalize_items = [(pid, d["pdf_path"], d["content_list_path"], d["middle_json_path"])
                       for pid, d in mineru_outputs.items()]
    if "normalize" in requested:
        to_run = normalize_items if not args.resume else [
            t for t in normalize_items if not state.already_ok(t[0], "normalize")
        ]
        results = _dispatch(
            "normalize", to_run,
            lambda chunk: stage_normalize.run_chunk(chunk, output_dir, dpi=args.dpi),
            use_slurm=args.slurm, cpu_partition=args.cpu_partition,
            gpu_partition=args.gpu_partition, chunk_size=args.chunk_size,
            logs_dir=logs_dir, overrides=overrides,
        )
        state.append_many(results)

    # ===== filter =====
    filter_items: list[tuple[str, Path, Path]] = []
    for paper_id in mineru_outputs:
        if state.already_ok(paper_id, "normalize"):
            norm_dir = output_dir / "normalized" / paper_id
            cl = norm_dir / f"{paper_id}_content_list.json"
            imgs = norm_dir / "page_images"
            filter_items.append((paper_id, cl, imgs))
    if "filter" in requested:
        to_run = filter_items if not args.resume else [
            t for t in filter_items if not state.already_ok(t[0], "filter")
        ]
        results = _dispatch(
            "filter", to_run,
            lambda chunk: stage_filter.run_chunk(chunk, min_body_pages=args.min_body_pages),
            use_slurm=args.slurm, cpu_partition=args.cpu_partition,
            gpu_partition=args.gpu_partition, chunk_size=args.chunk_size,
            logs_dir=logs_dir, overrides=overrides,
        )
        state.append_many(results)

    # ===== export =====
    if "export" in requested:
        # Build the per-paper item list using the latest state for body_pages
        body_pages_by_id = _read_body_pages(state)
        manifest_extras = {e.id: e.extra for e in entries}
        export_items = []
        for paper_id, cl, imgs in filter_items:
            if not state.already_ok(paper_id, "filter"):
                continue
            export_items.append({
                "id": paper_id,
                "content_list_path": cl,
                "images_dir": imgs,
                "body_pages": body_pages_by_id.get(paper_id, args.min_body_pages),
                "extra": manifest_extras.get(paper_id, {}),
            })
        # Export is fast and runs locally on the driver — no need for slurm
        text_entries, vision_entries = stage_export.build_entries(
            export_items, task_prompt=args.task_prompt
        )
        text_path, vision_path = stage_export.write_outputs(text_entries, vision_entries, output_dir)
        for it in export_items:
            state.append(StageResult(
                id=it["id"], stage="export", status="ok",
                output_path=str(text_path),
                extra={"vision_path": str(vision_path)},
            ))
        print(f"[paperprep] wrote {len(text_entries)} text + {len(vision_entries)} vision entries")

    _print_summary(state)
    return 0


def _lookup_output_path(state: StateLog, paper_id: str, stage: str) -> str | None:
    """Walk the JSONL log backwards to find the latest output_path for (id, stage)."""
    if not state.path.exists():
        return None
    found: str | None = None
    with open(state.path) as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("id") == paper_id and row.get("stage") == stage and row.get("status") == "ok":
                found = row.get("output_path") or found
    return found


def _find_first(directory: Path, pattern: str) -> Path | None:
    if not directory.is_dir():
        return None
    matches = list(directory.rglob(pattern))
    return matches[0] if matches else None


def _read_body_pages(state: StateLog) -> dict[str, int]:
    """Pull body_pages from the latest filter-stage row per paper."""
    out: dict[str, int] = {}
    if not state.path.exists():
        return out
    with open(state.path) as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("stage") == "filter" and row.get("status") == "ok":
                bp = (row.get("extra") or {}).get("body_pages")
                if isinstance(bp, int):
                    out[row["id"]] = bp
    return out


def _print_summary(state: StateLog) -> None:
    summary = state.summarize()
    if not summary:
        print("[paperprep] no state recorded")
        return
    print("\n[paperprep] per-stage summary:")
    for stage in ["compile", "page_count_filter", "mineru", "normalize", "filter", "export"]:
        if stage in summary:
            counts = summary[stage]
            parts = [f"{k}={v}" for k, v in sorted(counts.items())]
            print(f"  {stage:20s} {' '.join(parts)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="paperprep",
        description="Batch-process LaTeX dirs / PDFs through the arxiv-style "
                    "compile -> MinerU -> normalize -> ShareGPT pipeline.",
    )
    sub_parsers = parser.add_subparsers(dest="cmd", required=True)

    run = sub_parsers.add_parser("run", help="Run the end-to-end pipeline")
    src_g = run.add_mutually_exclusive_group(required=True)
    src_g.add_argument("--input-manifest", type=str,
                       help="JSONL manifest (one paper per line; see manifest.py)")
    src_g.add_argument("--input-dir", type=str,
                       help="Directory of LaTeX dirs (with --input-mode latex_dir) or PDFs (--input-mode pdf)")
    run.add_argument("--input-mode", choices=["latex_dir", "pdf"],
                     help="Required with --input-dir; ignored with --input-manifest")
    run.add_argument("--output-dir", required=True, type=str,
                     help="All per-stage artifacts written here")
    run.add_argument("--logs-dir", type=str, default=None,
                     help="Submitit log dir; default: <output-dir>/logs/<run_id>/")
    run.add_argument("--run-id", type=str, default=None,
                     help="Custom run-id; default: timestamp-uuid")

    # Stage selection + resume
    run.add_argument("--stages", type=str, default=",".join(ALL_STAGES),
                     help=f"Comma-separated subset of {ALL_STAGES}; default: all")
    run.add_argument("--no-resume", dest="resume", action="store_false", default=True,
                     help="Re-run papers regardless of prior state.jsonl status")

    # Slurm vs local
    run.add_argument("--slurm", action="store_true",
                     help="Submit each stage via submitit; otherwise run locally")
    run.add_argument("--gpu-partition", type=str, default=None,
                     help="SLURM partition for the MinerU stage")
    run.add_argument("--cpu-partition", type=str, default=None,
                     help="SLURM partition for compile, normalize, filter, export")
    run.add_argument("--chunk-size", type=int, default=50,
                     help="Papers per submitit task (mineru clamped to 20)")
    run.add_argument("--stage-overrides", type=str, default="",
                     help="Comma-separated stage.key=value (e.g. mineru.timeout_min=480)")

    # Stage-specific knobs
    run.add_argument("--compile-timeout", type=int, default=600,
                     help="Per-paper latexmk timeout (seconds)")
    run.add_argument("--texlive-bin", type=str, default=None,
                     help="Dir to prepend to PATH so the right pdflatex is found")
    run.add_argument("--max-pages", type=int, default=14,
                     help="Cap per-PDF MinerU pages")
    run.add_argument("--dpi", type=int, default=150,
                     help="Page-image DPI for vision rasterization")
    run.add_argument("--min-body-pages", type=int, default=6,
                     help="Drop papers with body_pages below this (arxiv default 6)")
    run.add_argument("--task-prompt", type=str, default=None,
                     help="Override SHAREGPT_USER_PROMPT_GENERIC")

    # ------------------------------------------------------------------
    # `paperprep serve` — long-lived HTTP server with persistent MinerU
    # ------------------------------------------------------------------
    serve = sub_parsers.add_parser(
        "serve", help="Run paperprep as a long-lived HTTP server "
                      "(persistent MinerU vLLM backend)",
    )
    serve.add_argument("--output-dir", required=True, type=str,
                       help="Per-request subdirs are created here; outputs persist after the response")
    serve.add_argument("--host", default="127.0.0.1",
                       help="Bind address; default 127.0.0.1 (use 0.0.0.0 only behind a trusted network)")
    serve.add_argument("--port", type=int, default=5000,
                       help="Flask port")
    serve.add_argument("--mineru-port", type=int, default=30000,
                       help="Port for the spawned mineru-vllm-server child")
    serve.add_argument("--mineru-max-pages", type=int, default=14,
                       help="Cap per-PDF MinerU pages")
    serve.add_argument("--min-body-pages", type=int, default=6,
                       help="Drop papers with body_pages below this (matches `paperprep run`)")
    serve.add_argument("--gpu-memory-utilization", type=float, default=0.7,
                       help="vLLM --gpu-memory-utilization passed to mineru-vllm-server")
    serve.add_argument("--texlive-bin", type=str, default=None,
                       help="Dir to prepend to PATH so the right pdflatex is found")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "run":
        if args.slurm and (not args.gpu_partition or not args.cpu_partition):
            print("ERROR: --slurm requires --gpu-partition and --cpu-partition",
                  file=sys.stderr)
            return 2
        if args.input_dir and not args.input_mode:
            print("ERROR: --input-dir requires --input-mode latex_dir|pdf",
                  file=sys.stderr)
            return 2
        from paper_anonymizer.core.constants import SHAREGPT_USER_PROMPT_GENERIC
        if args.task_prompt is None:
            args.task_prompt = SHAREGPT_USER_PROMPT_GENERIC
        return cmd_run(args)
    if args.cmd == "serve":
        # Lazy import: flask is an optional dep (`--extra serve`); don't
        # pull it in when the user just runs `paperprep run`.
        from paper_anonymizer.paperprep import serve as serve_mod
        return serve_mod.run_serve(
            Path(args.output_dir),
            host=args.host,
            port=args.port,
            mineru_port=args.mineru_port,
            mineru_max_pages=args.mineru_max_pages,
            min_body_pages=args.min_body_pages,
            gpu_memory_utilization=args.gpu_memory_utilization,
            texlive_bin=args.texlive_bin,
        )
    return 1


if __name__ == "__main__":
    sys.exit(main())
