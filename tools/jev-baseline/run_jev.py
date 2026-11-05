"""Zero-shot Jev baseline: ask TypeSafe's Jev whether each paper will be accepted.

Jev sees the same text as PaperScorer-Text. PaperScorer's user message is an instruction preamble followed by
the paper (title + body in markdown). We strip the preamble, send the paper as Jev's `state`, and ask one Noul
question that carries the same information as the preamble. The returned `noul` is P(accept).

Inputs are the sharegpt `data.json` files rebuilt from the Hugging Face dataset `skonan/PaperScorer-Text`
with `paperscorer-training-and-inference/scripts/reconstruction.py` (see README.md). Outputs are appended to
<out_dir>/<set>.jsonl, one row per paper; rows already scored are skipped on rerun.

    export TYPESAFE_API_KEY=...
    python run_jev.py --data_root ../../paperscorer-training-and-inference/data --sets all
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-1.13.0"  # pinned version used in the paper
TOKENS_PER_SEC = 80_000  # stay under the documented 100k tokens/s

# Balanced validation and test sets of the main results table (keys from the HF dataset's manifest.json).
_I = "iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_y25up"
_A = "arxiv_50_50_21k_text_wmetadata_filtered24480_y24up"
SETS = {
    "iclr_bal_val": f"{_I}_validation",
    "iclr_bal_test": f"{_I}_test",
    "arxiv_bal_val": f"{_A}_validation",
    "arxiv_bal_test": f"{_A}_test",
}

# PaperScorer's preambles (paperprep core/constants.py) and the matching Jev questions.
PREAMBLE = {
    "iclr": "I am giving you a paper. I want to predict its acceptance outcome at ICLR.\n"
            " - Your answer will either be: \\boxed{Accept} or \\boxed{Reject}\n"
            " - Note: ICLR generally has a ~30% acceptance rate",
    "arxiv": "I am giving you a paper submitted to a top machine-learning venue. Predict its acceptance outcome.\n"
             " - Your answer will either be: \\boxed{Accept} or \\boxed{Reject}\n"
             " - Note: typical top-tier ML venues have ~25-30% acceptance rates",
}
QUESTION = {
    "iclr": "This paper was submitted to ICLR. Will it be accepted? "
            "Note: ICLR generally has a ~30% acceptance rate.",
    "arxiv": "This paper was submitted to a top machine-learning venue. Will it be accepted? "
             "Note: typical top-tier ML venues have ~25-30% acceptance rates.",
}


def meta(row: dict) -> dict:
    return row.get("_metadata") or {}


def gold(row: dict) -> int:
    ans = meta(row).get("answer") or row["conversations"][-1]["value"]
    return 1 if "accept" in str(ans).lower() else 0


def paper_id(row: dict) -> str:
    m = meta(row)
    return str(m.get("submission_id") or m.get("arxiv_id"))


def paper_text(row: dict, domain: str) -> str:
    """The paper as PaperScorer sees it, without PaperScorer's instruction preamble."""
    t = next(x for x in row["conversations"] if x.get("from") in ("human", "user"))["value"]
    if not t.startswith(PREAMBLE[domain]):
        raise ValueError("unexpected preamble")
    return t[len(PREAMBLE[domain]):].lstrip()


def request_body(row: dict, domain: str) -> dict:
    return {"model": MODEL, "state": {"paper": paper_text(row, domain)},
            "questions": {"accepted": {"type": "noul", "instructions": QUESTION[domain]}}}


class TokenBucket:
    def __init__(self, rate: float):
        self.rate, self.tokens, self.t, self.lock = rate, rate, time.monotonic(), threading.Lock()

    def take(self, n: float):
        while True:
            with self.lock:
                now = time.monotonic()
                self.tokens = min(self.rate, self.tokens + (now - self.t) * self.rate)
                self.t = now
                if self.tokens >= n:
                    self.tokens -= n
                    return
                wait = (n - self.tokens) / self.rate
            time.sleep(wait)


def call(body: dict, key: str, bucket: TokenBucket, est_tokens: int) -> dict:
    data = json.dumps(body).encode()
    last = ""
    for attempt in range(8):
        bucket.take(min(est_tokens, TOKENS_PER_SEC))
        req = urllib.request.Request(URL, data=data, method="POST",
                                     headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}: {e.read()[:300].decode(errors='replace')}"
            if e.code in (429, 500, 502, 503, 504, 529):
                time.sleep(min(60, 2 ** attempt + random.random()))
                continue
            return {"error": last}
        except (urllib.error.URLError, TimeoutError) as e:
            last = str(e)
            time.sleep(min(60, 2 ** attempt + random.random()))
    return {"error": f"gave up after retries: {last}"}


def run_set(set_name: str, data_root: Path, out_dir: Path, limit: int | None, workers: int, key: str,
            bucket: TokenBucket):
    domain = set_name.split("_")[0]
    rows = json.loads((data_root / SETS[set_name] / "data.json").read_text())
    idx = list(range(len(rows)))
    if limit:
        random.Random(0).shuffle(idx)
        idx = sorted(idx[:limit])
    out = out_dir / f"{set_name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)["idx"] for line in open(out) if '"noul"' in line} if out.exists() else set()
    todo = [i for i in idx if i not in done]
    print(f"{set_name}: {len(idx)} papers, {len(idx) - len(todo)} done, {len(todo)} to run", flush=True)
    n_ok = n_err = 0

    def work(i):
        rec = {"idx": i, "paper_id": paper_id(rows[i]), "gold": gold(rows[i])}
        try:
            body = request_body(rows[i], domain)
        except ValueError as e:
            return dict(rec, error=str(e))
        resp = call(body, key, bucket, est_tokens=len(body["state"]["paper"]) // 4)
        if "error" in resp:
            return dict(rec, error=resp["error"])
        rec.update(noul=resp["answers"]["accepted"]["noul"], model=resp.get("model"),
                   input_tokens=resp.get("usage", {}).get("input_tokens"))
        return rec

    with ThreadPoolExecutor(workers) as pool, open(out, "a") as fh:
        for fut in as_completed([pool.submit(work, i) for i in todo]):
            rec = fut.result()
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            n_ok, n_err = (n_ok + 1, n_err) if "noul" in rec else (n_ok, n_err + 1)
    print(f"{set_name}: ok {n_ok}, errors {n_err}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", type=Path, required=True, help="reconstruction.py output dir (holds <key>/data.json)")
    ap.add_argument("--out_dir", type=Path, default=Path(__file__).resolve().parent / "outputs")
    ap.add_argument("--sets", default="all", help=f"comma list of {sorted(SETS)} or 'all'")
    ap.add_argument("--limit", type=int, default=None, help="random subset per set (seed 0), for a pilot")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        sys.exit("TYPESAFE_API_KEY is not set")
    bucket = TokenBucket(TOKENS_PER_SEC)
    for s in (list(SETS) if a.sets == "all" else a.sets.split(",")):
        run_set(s, a.data_root, a.out_dir, a.limit, a.workers, key, bucket)


if __name__ == "__main__":
    main()
