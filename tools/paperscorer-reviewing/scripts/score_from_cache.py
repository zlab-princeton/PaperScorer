"""Stage B of the model-grid eval: score every cached paper row for ONE model.

Reads the prep cache (build_prep_cache.py output), POSTs each paper's
<modality>.json row to a paperlens-serve already loaded with the target ckpt,
writes one jsonl of {paper_id, group, order, date, subject, p_accept, decision}.

Usage:
  python score_from_cache.py --cache-dir grid/prepped --modality vision \
      --paperlens-url http://127.0.0.1:8028 \
      --model-label arxivS-7b-vision --out grid/scores/arxivS-7b-vision.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--modality", choices=["text", "vision"], required=True)
    ap.add_argument("--paperlens-url", required=True)
    ap.add_argument("--model-label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout", type=float, default=600.0)
    args = ap.parse_args()

    cache = Path(args.cache_dir).resolve()
    out = Path(args.out).resolve(); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("")

    try:
        requests.get(args.paperlens_url.rstrip("/") + "/health", timeout=5).raise_for_status()
    except Exception as e:
        print(f"ERROR: paperlens not healthy: {e}", file=sys.stderr); return 2

    paper_dirs = sorted([d for d in cache.iterdir() if d.is_dir() and not d.name.startswith("_")])
    n_ok = n_skip = 0
    for d in paper_dirs:
        meta_f = d / "meta.json"
        if not meta_f.exists():
            continue
        meta = json.loads(meta_f.read_text())
        if meta.get("state") != "ok":
            continue
        row_f = d / f"{args.modality}.json"
        if not row_f.exists():
            n_skip += 1
            continue  # e.g. vision model but paper had no images
        row = json.loads(row_f.read_text())
        rec = {"model": args.model_label, "modality": args.modality,
               "paper_id": meta["paper_id"], "group": meta.get("group"),
               "order": meta.get("order"), "date": meta.get("date"),
               "subject": meta.get("subject")}
        try:
            r = requests.post(args.paperlens_url.rstrip("/") + "/score",
                              json={"papers": [row]}, timeout=args.timeout)
            r.raise_for_status()
            sc = r.json()["scores"][0]
            pa = float(sc["p_accept"])
            rec.update(state="done", p_accept=round(pa, 4),
                       decision="Accept" if pa >= 0.5 else "Reject",
                       logp_accept=sc.get("logp_accept"), logp_reject=sc.get("logp_reject"))
            n_ok += 1
        except Exception as e:
            rec.update(state="error", error=str(e))
        with out.open("a") as f:
            f.write(json.dumps(rec) + "\n")
    print(f"[score_from_cache] {args.model_label} ({args.modality}): {n_ok} scored, {n_skip} skipped(no row) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
