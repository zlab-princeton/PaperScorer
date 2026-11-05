"""Score a paper across its git commits with PaperLens.

For each of the last N commits of a LaTeX paper repo:
  1. `git archive <sha>` -> isolated temp tree (never touches the working tree)
  2. POST paperprep-serve /prepare {type: latex_dir, main_tex} -> paperprep
     anonymizes (authors->Anonymous, strips github/urls/acks) + compiles
     (latexmk, or pdflatex+bibtex fallback) + mineru + normalize + filter +
     export sharegpt
  3. POST paperlens-serve /score on the vision row -> p_accept
  4. record (sha, date, subject, p_accept, decision, body_pages)

Writes results.jsonl (one row per commit, oldest->newest). Per-commit failures
(compile error, filter reject) are recorded, not fatal.

NOTE: scores in the lab's own arXiv template (zlab.cls). Format bias is a
constant offset across commits, so the *trajectory* is the trustworthy signal,
not the absolute p_accept (see paper-commit-tracker plan).

Usage:
  python score_commits.py --repo /scratch/.../PaperLensArXivRelease \
      --n-commits 10 --out-dir logs/commit_track \
      [--paperprep-url http://127.0.0.1:8004] [--paperlens-url http://127.0.0.1:8002] \
      [--main-tex main.tex] [--modality vision]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import requests


def select_commits(repo: Path, n: int | None, from_commit: str | None) -> list[dict]:
    """Return commits oldest->newest as {sha, ts, subject}.

    --from-commit <sha>: every commit from <sha> through HEAD (inclusive).
    else: the last n commits.
    """
    if from_commit:
        rev = f"{from_commit}^..HEAD"   # inclusive of from_commit
        log_args = [rev]
    else:
        log_args = [f"-{n}"]
    out = subprocess.run(
        ["git", "-C", str(repo), "log", *log_args, "--format=%H%x1f%ct%x1f%s"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()
    commits = []
    for line in out:
        sha, ts, subject = line.split("\x1f", 2)
        commits.append({"sha": sha, "ts": int(ts), "subject": subject})
    commits.reverse()  # oldest -> newest
    return commits


def archive_commit(repo: Path, sha: str, dest: Path) -> None:
    """Extract the repo tree at <sha> into dest (created fresh)."""
    dest.mkdir(parents=True, exist_ok=True)
    p = subprocess.Popen(["git", "-C", str(repo), "archive", sha], stdout=subprocess.PIPE)
    subprocess.run(["tar", "-x", "-C", str(dest)], stdin=p.stdout, check=True)
    p.wait()
    if p.returncode != 0:
        raise RuntimeError(f"git archive {sha[:8]} failed")


def prepare(paperprep_url: str, sha: str, tree: Path, main_tex: str, timeout: float) -> dict:
    """POST a latex_dir to paperprep-serve /prepare."""
    payload = {
        "request_id": sha[:12],
        "papers": [{"id": sha[:12], "type": "latex_dir", "path": str(tree), "main_tex": main_tex}],
    }
    r = requests.post(paperprep_url.rstrip("/") + "/prepare", json=payload, timeout=timeout)
    if r.status_code >= 500:
        raise RuntimeError(f"paperprep /prepare {r.status_code}: {r.text[:300]}")
    r.raise_for_status()
    return r.json()


def load_sharegpt_row(prepare_body: dict, modality: str) -> dict | None:
    key = f"sharegpt_{modality}_path"
    p = prepare_body.get(key)
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
    return row


def score(paperlens_url: str, row: dict, timeout: float) -> dict:
    r = requests.post(paperlens_url.rstrip("/") + "/score", json={"papers": [row]}, timeout=timeout)
    r.raise_for_status()
    body = r.json()
    if not body.get("scores"):
        raise RuntimeError(f"paperlens /score returned no scores: {body}")
    return body["scores"][0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--n-commits", type=int, default=10)
    ap.add_argument("--from-commit", default=None,
                    help="score every commit from this sha through HEAD (inclusive); overrides --n-commits")
    ap.add_argument("--num-slices", type=int, default=1,
                    help="split the selected commit window into this many contiguous slices")
    ap.add_argument("--slice-index", type=int, default=0,
                    help="which slice (0-based) this run processes")
    ap.add_argument("--results-name", default="results.jsonl",
                    help="output filename under --out-dir (use distinct names for parallel slices)")
    ap.add_argument("--paperprep-url", default="http://127.0.0.1:8004")
    ap.add_argument("--paperlens-url", default="http://127.0.0.1:8002")
    ap.add_argument("--main-tex", default="main.tex")
    ap.add_argument("--modality", choices=["text", "vision"], default="vision")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--work-dir", default=None, help="where to extract commit trees (default <out>/trees)")
    ap.add_argument("--timeout", type=float, default=900.0)
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    out_dir = Path(args.out_dir).resolve(); out_dir.mkdir(parents=True, exist_ok=True)
    work = Path(args.work_dir).resolve() if args.work_dir else out_dir / "trees"
    results_path = out_dir / args.results_name
    results_path.write_text("")

    # Health pre-flight
    for name, url, path in [("paperprep", args.paperprep_url, "/healthz"),
                            ("paperlens", args.paperlens_url, "/health")]:
        try:
            requests.get(url.rstrip("/") + path, timeout=5).raise_for_status()
        except Exception as e:
            print(f"ERROR: {name} not healthy at {url}{path}: {e}", file=sys.stderr)
            return 2

    commits = select_commits(repo, args.n_commits, args.from_commit)
    # Keep a stable global order index BEFORE slicing, so parallel slices
    # produce rows that merge back into one chronological trajectory.
    for gi, c in enumerate(commits):
        c["order"] = gi
    if args.num_slices > 1:
        n = len(commits)
        per = -(-n // args.num_slices)  # ceil
        lo = args.slice_index * per
        hi = min(lo + per, n)
        commits = commits[lo:hi]
        print(f"[score_commits] slice {args.slice_index}/{args.num_slices}: "
              f"global commits [{lo}:{hi}] of {n}")
    print(f"[score_commits] {len(commits)} commits this run, modality={args.modality}, repo={repo}")

    for i, c in enumerate(commits):
        sha, subject = c["sha"], c["subject"]
        date = time.strftime("%Y-%m-%d", time.localtime(c["ts"]))
        short = sha[:8]
        rec = {"order": c["order"], "sha": sha, "short": short, "date": date, "subject": subject,
               "state": "pending"}
        print(f"\n[{i+1}/{len(commits)} | global {c['order']}] {short} {date}  {subject[:60]}", flush=True)
        try:
            tree = work / short
            archive_commit(repo, sha, tree)
            t0 = time.time()
            body = prepare(args.paperprep_url, sha, tree, args.main_tex, args.timeout)
            papers = body.get("papers") or []
            p0 = papers[0] if papers else {}
            if p0.get("status") != "ok":
                rec.update({"state": "prep_failed", "error": f"{p0.get('status')}: {p0.get('error')}"})
                print(f"  prep failed: {rec['error']}", flush=True)
            else:
                row = load_sharegpt_row(body, args.modality)
                if row is None:
                    rec.update({"state": "no_sharegpt", "error": f"no {args.modality} row"})
                else:
                    sc = score(args.paperlens_url, row, args.timeout)
                    pa = float(sc["p_accept"])
                    rec.update({
                        "state": "done",
                        "p_accept": round(pa, 4),
                        "decision": "Accept" if pa >= 0.5 else "Reject",
                        "logp_accept": sc.get("logp_accept"),
                        "logp_reject": sc.get("logp_reject"),
                        "body_pages": (p0.get("body_pages") or (row.get("_metadata") or {}).get("body_pages")),
                        "prep_elapsed_s": round(time.time() - t0, 1),
                        "paperprep_output_dir": body.get("output_dir"),
                    })
                    print(f"  -> {rec['decision']}  p_accept={rec['p_accept']}", flush=True)
        except Exception as e:
            rec.update({"state": "error", "error": str(e)})
            print(f"  ERROR: {e}", flush=True)
        with results_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")

    print(f"\n[score_commits] DONE -> {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
