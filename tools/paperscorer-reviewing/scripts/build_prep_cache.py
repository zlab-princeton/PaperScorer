"""Stage A of the model-grid eval: prep every paper ONCE through paperprep,
cache its text + vision sharegpt rows. Decouples the expensive compile/OCR
from scoring so the 10-model grid never re-preps.

Papers:
  - git commits of a LaTeX repo  -> type=latex_dir (paperprep anonymizes+compiles)
  - standalone PDFs              -> type=pdf (already-anonymized; skip compile)

Cache layout (idempotent — re-runs skip cached papers):
  <cache>/<paper_id>/meta.json      {paper_id, group, order, date, subject, source}
  <cache>/<paper_id>/text.json      sharegpt text row (gpt turn appended)
  <cache>/<paper_id>/vision.json    sharegpt vision row (may be absent if no images)
  <cache>/<paper_id>/FAILED         (touch file) if prep failed; meta.json holds error

Usage (slice-parallel):
  python build_prep_cache.py --cache-dir grid/prepped \
    --repo /scratch/.../PaperLensArXivRelease --from-commit f48b7f8 \
    --test-pdfs-root /scratch/.../test_pdfs \
    --extra-pdf neurips=/scratch/.../neurips_2026.pdf \
    --paperprep-url http://127.0.0.1:8024 \
    --num-slices 3 --slice-index 0
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import zlib
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_commits import archive_commit  # reuse git-archive helper


def gather_papers(args) -> list[dict]:
    """Build the global paper list (stable order): commits first, then PDFs."""
    papers: list[dict] = []
    # commits (latex_dir)
    if args.repo and args.from_commit:
        repo = Path(args.repo).resolve()
        log = subprocess.run(
            ["git", "-C", str(repo), "log", f"{args.from_commit}^..HEAD",
             "--format=%H%x1f%ct%x1f%s"],
            capture_output=True, text=True, check=True,
        ).stdout.strip().splitlines()
        rows = []
        for line in log:
            sha, ts, subject = line.split("\x1f", 2)
            rows.append((sha, int(ts), subject))
        rows.reverse()  # oldest -> newest
        for sha, ts, subject in rows:
            papers.append({
                "paper_id": "commit_" + sha[:8], "group": "commit",
                "type": "latex_dir", "sha": sha, "repo": str(repo),
                "main_tex": args.main_tex,
                "date": time.strftime("%Y-%m-%d", time.localtime(ts)),
                "subject": subject,
            })
    # test-pdfs tree
    if args.test_pdfs_root:
        root = Path(args.test_pdfs_root).resolve()
        for pdf in sorted(root.rglob("*.pdf")):
            grp = pdf.relative_to(root).parts[0] if len(pdf.relative_to(root).parts) > 1 else "root"
            papers.append({
                "paper_id": f"{grp}__{pdf.stem}".replace(" ", "_").replace("/", "_"),
                "group": grp, "type": "pdf", "path": str(pdf),
                "date": "", "subject": pdf.name,
            })
    # extra PDFs (label=path)
    for spec in args.extra_pdf or []:
        label, _, path = spec.partition("=")
        papers.append({"paper_id": label, "group": "extra", "type": "pdf",
                       "path": path, "date": "", "subject": Path(path).name})
    for gi, p in enumerate(papers):
        p["order"] = gi
    return papers


def prepare(url: str, paper: dict, timeout: float, work: Path) -> dict:
    pid = paper["paper_id"]
    if paper["type"] == "latex_dir":
        tree = work / pid
        archive_commit(Path(paper["repo"]), paper["sha"], tree)
        item = {"id": pid, "type": "latex_dir", "path": str(tree), "main_tex": paper["main_tex"]}
    else:
        item = {"id": pid, "type": "pdf", "path": paper["path"]}
    r = requests.post(url.rstrip("/") + "/prepare",
                      json={"request_id": pid, "papers": [item]}, timeout=timeout)
    if r.status_code >= 500:
        raise RuntimeError(f"paperprep {r.status_code}: {r.text[:300]}")
    r.raise_for_status()
    return r.json()


def cache_row(body: dict, modality: str, pdir: Path) -> dict | None:
    """Build the sharegpt row for `modality`. For vision, COPY the page PNGs
    into <pdir>/page_images/ and rewrite the row's `images` to point there, so
    the cache is self-contained (we then delete the bulky paperprep work tree).
    """
    import shutil
    p = body.get(f"sharegpt_{modality}_path")
    if not p or not Path(p).exists():
        return None
    rows = json.loads(Path(p).read_text())
    if not rows:
        return None
    row = dict(rows[0])
    convs = list(row.get("conversations", []))
    if not any(c.get("from") == "gpt" for c in convs):
        convs.append({"from": "gpt", "value": "Outcome: \\boxed{Accept}"})
        row["conversations"] = convs
    if modality == "vision" and row.get("images"):
        img_dst = pdir / "page_images"
        img_dst.mkdir(parents=True, exist_ok=True)
        new_imgs = []
        for src in row["images"]:
            sp = Path(src)
            dp = img_dst / sp.name
            try:
                shutil.copyfile(sp, dp)
                new_imgs.append(str(dp))
            except Exception:
                pass  # missing page -> drop it
        row["images"] = new_imgs
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--repo", default=None)
    ap.add_argument("--from-commit", default=None)
    ap.add_argument("--main-tex", default="main.tex")
    ap.add_argument("--test-pdfs-root", default=None)
    ap.add_argument("--extra-pdf", action="append", default=[], help="label=/abs/path.pdf")
    ap.add_argument("--paperprep-url", default="http://127.0.0.1:8024")
    ap.add_argument("--num-slices", type=int, default=1)
    ap.add_argument("--slice-index", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=900.0)
    args = ap.parse_args()

    cache = Path(args.cache_dir).resolve(); cache.mkdir(parents=True, exist_ok=True)
    work = cache / "_trees"; work.mkdir(exist_ok=True)

    try:
        requests.get(args.paperprep_url.rstrip("/") + "/healthz", timeout=5).raise_for_status()
    except Exception as e:
        print(f"ERROR: paperprep not healthy: {e}", file=sys.stderr); return 2

    papers = gather_papers(args)
    if args.num_slices > 1:
        papers = [p for p in papers if zlib.crc32(p["paper_id"].encode()) % args.num_slices == args.slice_index]
    print(f"[prep-cache] slice {args.slice_index}/{args.num_slices}: {len(papers)} papers")

    for i, p in enumerate(papers):
        pid = p["paper_id"]
        pdir = cache / pid
        if (pdir / "meta.json").exists() and not (pdir / "FAILED").exists():
            print(f"[{i+1}/{len(papers)}] {pid}  (cached, skip)"); continue
        pdir.mkdir(parents=True, exist_ok=True)
        print(f"[{i+1}/{len(papers)}] {pid}  ({p['type']}, {p['group']})", flush=True)
        meta = {k: p[k] for k in ("paper_id", "group", "order", "date", "subject", "type")}
        try:
            body = prepare(args.paperprep_url, p, args.timeout, work)
            pr = (body.get("papers") or [{}])[0]
            if pr.get("status") != "ok":
                meta["state"] = "prep_failed"; meta["error"] = f"{pr.get('status')}: {pr.get('error')}"
                (pdir / "FAILED").write_text(meta["error"])
                print(f"   prep_failed: {meta['error'][:80]}", flush=True)
            else:
                trow = cache_row(body, "text", pdir); vrow = cache_row(body, "vision", pdir)
                if trow: (pdir / "text.json").write_text(json.dumps(trow))
                if vrow: (pdir / "vision.json").write_text(json.dumps(vrow))
                meta["state"] = "ok"
                meta["has_text"] = bool(trow); meta["has_vision"] = bool(vrow)
                meta["body_pages"] = pr.get("body_pages")
                if (pdir / "FAILED").exists(): (pdir / "FAILED").unlink()
                print(f"   ok  text={bool(trow)} vision={bool(vrow)}", flush=True)
            # Inode hygiene: drop this paper's bulky paperprep output subtree
            # (hundreds of MinerU crops) now that the page PNGs we need are
            # copied into the cache. Also drop the git-archive tree if latex.
            import shutil
            od = body.get("output_dir")
            if od and Path(od).exists():
                shutil.rmtree(od, ignore_errors=True)
            if p["type"] == "latex_dir":
                shutil.rmtree(work / pid, ignore_errors=True)
        except Exception as e:
            meta["state"] = "error"; meta["error"] = str(e)
            (pdir / "FAILED").write_text(str(e))
            import shutil
            if p["type"] == "latex_dir":
                shutil.rmtree(work / pid, ignore_errors=True)
            print(f"   ERROR: {e}", flush=True)
        (pdir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"[prep-cache] slice {args.slice_index} done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
