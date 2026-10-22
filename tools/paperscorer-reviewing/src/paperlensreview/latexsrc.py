"""LaTeX-source + git helpers for the reviewing UI's "review a local LaTeX dir"
input mode.

Pure stdlib + the ``git`` / ``tar`` CLIs; no paperprep import (that lives behind
the paperprep-serve HTTP boundary). Conventions match the offline batch scripts
(``scripts/score_commits.py``): commits are git-archived into an isolated tree
(never touching the working tree), short shas are ``sha[:8]``, dates come from
the commit timestamp ``%ct``.

History selection (see README-prod-git-run.md / interactive UI):
  * default window = the last N commits touching ``*.tex`` (N=50);
  * per-commit churn = added+deleted lines in ``*.tex`` only;
  * surface the mean churn (reference) and a tunable percentile cutoff
    (default 50 = median) with the value at that percentile;
  * commits with ``churn >= cutoff`` are pre-selected; the UI's slider
    lets the user re-tune the percentile client-side without re-querying.
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

_SEP = "\x1f"  # unit-separator: safe field delimiter inside commit subjects


# ---------------------------------------------------------------------------
# main.tex / entrypoint detection (mirrors paperprep find_main_tex; the UI uses
# this only to PRE-FILL the field -- paperprep re-detects authoritatively).
# ---------------------------------------------------------------------------

def find_main_tex(src_dir: Path) -> str | None:
    """Best-effort entrypoint guess, returned relative to ``src_dir``.

    00README.json ``toplevelfile`` -> first ``*.tex`` containing
    ``\\documentclass`` (shallowest path wins). Returns None if nothing matches.
    """
    import json
    readme = src_dir / "00README.json"
    if readme.is_file():
        try:
            data = json.loads(readme.read_text(encoding="utf-8", errors="replace"))
            for entry in (data.get("sources") or []):
                top = entry.get("filename") or entry.get("toplevelfile")
                if top and top.endswith(".tex") and (src_dir / top).is_file():
                    return top
        except Exception:
            pass
    for tex in sorted(src_dir.rglob("*.tex"), key=lambda p: (len(p.parts), str(p))):
        try:
            if "\\documentclass" in tex.read_text(encoding="utf-8", errors="replace"):
                return str(tex.relative_to(src_dir))
        except Exception:
            continue
    return None


def list_tex_files(src_dir: Path, limit: int = 200) -> list[str]:
    """All ``*.tex`` under ``src_dir`` (relative, shallowest-first), capped."""
    out: list[str] = []
    for tex in sorted(src_dir.rglob("*.tex"), key=lambda p: (len(p.parts), str(p))):
        out.append(str(tex.relative_to(src_dir)))
        if len(out) >= limit:
            break
    return out


# ---------------------------------------------------------------------------
# git plumbing
# ---------------------------------------------------------------------------

def _git(repo: Path, *args: str, check: bool = True) -> str:
    r = subprocess.run(["git", "-C", str(repo), *args],
                       capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {r.stderr.strip()[:300]}")
    return r.stdout


def is_git_repo(path: Path) -> bool:
    try:
        return _git(path, "rev-parse", "--is-inside-work-tree", check=False).strip() == "true"
    except Exception:
        return False


def git_toplevel(path: Path) -> Path | None:
    try:
        out = _git(path, "rev-parse", "--show-toplevel", check=False).strip()
        return Path(out) if out else None
    except Exception:
        return None


def resolve_commit(repo: Path, ref: str) -> str | None:
    """Resolve a ref/sha to its full commit sha, or None if it doesn't exist."""
    out = _git(repo, "rev-parse", "--verify", f"{ref}^{{commit}}", check=False).strip()
    return out or None


def working_tree_dirty(repo: Path) -> bool:
    """True if there are uncommitted changes (tracked) in the working tree."""
    return bool(_git(repo, "status", "--porcelain", check=False).strip())


def archive_commit(repo: Path, sha: str, dest: Path) -> None:
    """Extract the repo tree at ``sha`` into ``dest`` (tracked files only),
    without touching the working tree or index. Mirrors score_commits.py."""
    dest.mkdir(parents=True, exist_ok=True)
    p = subprocess.Popen(["git", "-C", str(repo), "archive", sha], stdout=subprocess.PIPE)
    subprocess.run(["tar", "-x", "-C", str(dest)], stdin=p.stdout, check=True)
    p.wait()
    if p.returncode != 0:
        raise RuntimeError(f"git archive {sha[:8]} failed")


# ---------------------------------------------------------------------------
# commit history + .tex churn
# ---------------------------------------------------------------------------

def _percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile (pure python; no numpy dependency)."""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    rank = (q / 100.0) * (len(s) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(s) - 1)
    return float(s[lo] + (s[hi] - s[lo]) * (rank - lo))


def _parse_numstat(out: str) -> list[dict]:
    """Parse ``git log --numstat --format=__C__%H\\x1f%ct\\x1f%an\\x1f%s``.

    Sums added+deleted across the (path-filtered) numstat rows per commit.
    Binary files report '-' for add/del and are skipped.
    """
    commits: list[dict] = []
    cur: dict | None = None
    for line in out.splitlines():
        if line.startswith("__C__"):
            if cur is not None:
                commits.append(cur)
            sha, ct, author, subject = line[5:].split(_SEP, 3)
            epoch = int(ct)
            cur = {
                "sha": sha,
                "short": sha[:8],
                "epoch": epoch,
                "date": time.strftime("%Y-%m-%d", time.localtime(epoch)),
                "author": author,
                "subject": subject,
                "churn": 0,
                "files": 0,
            }
        elif line.strip() and cur is not None:
            parts = line.split("\t")
            if len(parts) >= 3:
                added, deleted = parts[0], parts[1]
                if added != "-":
                    cur["churn"] += int(added)
                if deleted != "-":
                    cur["churn"] += int(deleted)
                cur["files"] += 1
    if cur is not None:
        commits.append(cur)
    return commits


def last_commits_with_tex_churn(repo: Path, n: int = 50, pct: float = 50.0) -> dict:
    """The last ``n`` commits touching ``*.tex``, oldest->newest, with per-commit
    ``.tex`` churn and a tunable selection threshold.

    ``pct`` is the percentile cutoff (default 50 = median): each commit gets
    ``above_pct = churn >= pct_churn``. Equal-to-threshold commits are
    included (``>=``, not strict). If every commit ends up below (degenerate),
    pre-select all.

    Returns ``{commits, mean_churn, pct, pct_churn, n}``. The UI receives this
    object, renders a percentile slider, and recomputes the selection
    client-side on the fly (no server round-trip per slider drag).
    """
    fmt = f"__C__%H{_SEP}%ct{_SEP}%an{_SEP}%s"
    out = _git(repo, "log", f"-{int(n)}", "--numstat", f"--format={fmt}", "--", "*.tex")
    commits = _parse_numstat(out)
    commits.reverse()  # oldest -> newest (HEAD anchored at the right of the graph)

    churns = [c["churn"] for c in commits]
    mean = sum(churns) / len(churns) if churns else 0.0
    p_value = _percentile(churns, pct)
    for c in commits:
        c["above_pct"] = c["churn"] >= p_value
    if commits and not any(c["above_pct"] for c in commits):
        # degenerate (all-zero etc.) -> nothing crossed the cutoff; pre-select all
        for c in commits:
            c["above_pct"] = True

    return {"commits": commits, "mean_churn": mean,
            "pct": pct, "pct_churn": p_value, "n": len(commits)}
