"""arxiv.org source fetch + extract for the /submit_arxiv flow.

Conventions mirror ``scripts/download_arxiv_sources.py`` on the
``SachinKonan/AutoReviewer:arXiv-papers`` branch:
  * URL = ``https://arxiv.org/src/{id}``
  * 404 is permanent (paper has no source bundle) — surface immediately.
  * 429 / 5xx / network errors retry with exponential backoff (2, 4, 8, 16, 32s).
  * Streamed download to ``<dest>.tmp`` then atomic rename.
  * Same arxiv-source shapes that ``paperprep.core.latex_compile.extract_source``
    handles: a gzipped tar (multi-file project) or a single gzipped ``.tex``.
"""
from __future__ import annotations

import gzip
import logging
import re
import tarfile
import time
from pathlib import Path

import requests


log = logging.getLogger(__name__)

# arxiv id shapes:
#   2305.00838 / 2305.00838v2          (new identifier)
#   cs.AI/0701234 / hep-th/0701234v3   (old identifier)
_NEW_ID = re.compile(r"^(\d{4}\.\d{4,5})(v\d+)?$")
_OLD_ID = re.compile(r"^([a-z\-]+(?:\.[A-Za-z]+)?/\d{7})(v\d+)?$")

# Extract id from full arxiv URLs (abs / pdf / src), tolerant of querystrings.
_URL_PATTERNS = [
    re.compile(r"arxiv\.org/(?:abs|pdf|src)/(\S+?)(?:[?#]|$)"),
]


class ArxivError(Exception):
    """Permanent failure (no retry) — 404 or malformed id."""


def normalize_id(raw: str) -> str:
    """Accept a bare arxiv id or any arxiv.org URL; return the canonical id.

    Trailing version (``v2``) is preserved if present so the caller can request
    a specific snapshot; pass a bare id for the latest version.
    """
    s = raw.strip()
    for pat in _URL_PATTERNS:
        m = pat.search(s)
        if m:
            s = m.group(1)
            break
    if s.lower().endswith(".pdf"):
        s = s[:-4]
    if _NEW_ID.match(s) or _OLD_ID.match(s):
        return s
    raise ArxivError(f"not a valid arxiv id or URL: {raw!r}")


def download_source(arxiv_id: str, dest: Path, *,
                    timeout: int = 60,
                    max_attempts: int = 5,
                    user_agent: str = "paperlens-reviewing/0.1 (mailto:sachinkonan480@gmail.com)") -> Path:
    """Stream ``https://arxiv.org/src/<arxiv_id>`` to ``dest`` atomically.

    Retries up to ``max_attempts`` times with exponential backoff (2,4,8,16,32s).
    Raises ``ArxivError`` on 404 (permanent) and after exhausting retries.
    """
    url = f"https://arxiv.org/src/{arxiv_id}"
    headers = {"User-Agent": user_agent}
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")

    last_err: Exception | None = None
    for attempt in range(max_attempts):
        try:
            with requests.get(url, timeout=timeout, headers=headers,
                              stream=True, allow_redirects=True) as r:
                if r.status_code == 404:
                    raise ArxivError(f"arxiv {arxiv_id!r} has no source (404)")
                if r.status_code == 429 or 500 <= r.status_code < 600:
                    raise _Transient(f"HTTP {r.status_code} from arxiv.org")
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=65536):
                        if chunk:
                            f.write(chunk)
            tmp.rename(dest)
            return dest
        except ArxivError:
            raise
        except (_Transient, requests.RequestException) as e:
            last_err = e
            if attempt < max_attempts - 1:
                delay = min(2.0 * (2 ** attempt), 60.0)
                log.warning("arxiv download attempt %d/%d failed (%s); retrying in %.0fs",
                            attempt + 1, max_attempts, e, delay)
                time.sleep(delay)
    raise ArxivError(f"giving up on {arxiv_id} after {max_attempts} attempts: {last_err}")


class _Transient(Exception):
    """Internal retryable signal — 429 / 5xx / network glitch."""


def extract_source(gz_path: Path, dest_dir: Path) -> bool:
    """Extract an arxiv source bundle into ``dest_dir``. Returns True on success.

    Two shapes (mirrors paperprep.core.latex_compile.extract_source):
      * gzipped tar (most papers) — use tarfile, ``filter="data"`` for safety.
      * single gzipped ``.tex`` (rare) — decompress to ``<stem>.tex``.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(str(gz_path)) as tf:
            tf.extractall(str(dest_dir), filter="data")
        return True
    except tarfile.ReadError:
        pass
    except Exception:
        return False
    try:
        with gzip.open(str(gz_path), "rb") as gz:
            content = gz.read()
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            text = content.decode("latin-1", errors="replace")
        (dest_dir / f"{gz_path.stem}.tex").write_text(text, encoding="utf-8")
        return True
    except Exception:
        return False
