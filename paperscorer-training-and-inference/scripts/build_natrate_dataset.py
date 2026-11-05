#!/usr/bin/env python3
"""
Build a per-conference natural-acceptance-rate-adjusted variant of the mini
arxiv dataset by drawing additional rejects from the big (balanced_per_venue)
dataset's pool.

Constraints:
  - Same total sample count per split as mini.
  - Same per-(split, venue) count as mini (so per-conference proportions are
    preserved — only the accept/reject mix per venue changes).
  - Sampling is deterministic (fixed seed).

For each (split, venue) cell:
    target_n = mini count
    target_a = round(target_n * accept_rate(venue))
    target_r = target_n - target_a
    accepts: random sample of target_a from mini's accepts in this cell.
    rejects: take all mini's rejects in this cell first, then top up from
             (big − mini) rejects in this cell until we have target_r.

Writes 3 datasets:
  arxiv_natrate_21k_text_wmetadata_filtered24480_{train,validation,test}
and registers them in data/dataset_info.json.
"""
from __future__ import annotations
import pathlib
import json
import random
from collections import defaultdict
from pathlib import Path

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"
SEED = 42

ACCEPT_RATE = {
    # Whole-percentage rounding of openaccept.org multi-year averages.
    # ACL/EMNLP/NAACL are rolled up into "acl_family" because the source data
    # treats them as one balanced (50/50) group — emnlp+naacl have no public
    # rejects, but acl has enough rejects to balance the trio at the group level.
    # Rate for acl_family is the simple average of (21.15, 21.49, 23.55)% = 22%.
    "aaai": 0.20,
    "acl_family": 0.22,
    "cvpr": 0.24,
    "icml": 0.25,
    "neurips": 0.26,
    "iccv": 0.26,
    "eccv": 0.28,
    "aistats": 0.29,
    "colm": 0.30,
    "iclr": 0.30,  # override (raw avg 31.05%)
    "corl": 0.39,
}

# Map raw venue -> group. Anything not listed here uses the venue itself as group.
VENUE_GROUP = {
    "acl": "acl_family",
    "emnlp": "acl_family",
    "naacl": "acl_family",
}


def group_of(v: str) -> str:
    return VENUE_GROUP.get(v, v)

SPLITS = ["train", "validation", "test"]


def get_prefixes(modality: str, source: str = "mini") -> tuple[str, str, str]:
    """Return (mini_prefix, big_prefix, out_prefix) for the chosen modality + source.

    source="mini": existing behavior — preserve mini per-venue counts, top up rejects from big.
    source="big":  natrate-per-venue — keep all rejects in big, subsample accepts to hit rate.
                   mini is unused (fed as same path so the cell loader doesn't blow up).
    """
    mini = f"arxiv_50_50_21k_{modality}_wmetadata_filtered24480"
    big = f"arxiv_50_50_balanced_per_venue_{modality}_wmetadata_filtered24480"
    if source == "mini":
        out = f"arxiv_natrate_21k_{modality}_wmetadata_filtered24480"
    elif source == "big":
        out = f"arxiv_natrate_balanced_per_venue_{modality}_wmetadata_filtered24480"
    else:
        raise ValueError(f"Unknown source: {source!r}")
    return (mini, big, out)


def venue(s):
    m = s.get("_metadata", {})
    return (m.get("pl_venue") or m.get("venue") or "?").lower()


def is_accept(s):
    return s["_metadata"]["answer"] == "Accept"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", choices=["text", "vision"], required=True)
    ap.add_argument("--source", choices=["mini", "big"], default="mini",
                    help="mini = preserve 21k per-venue counts (top up rejects from big); "
                         "big = keep all rejects in balanced_per_venue, subsample accepts.")
    args = ap.parse_args()

    MINI_PREFIX, BIG_PREFIX, OUT_PREFIX = get_prefixes(args.modality, args.source)
    print(f"Building natrate dataset for modality={args.modality} source={args.source}")
    print(f"  mini  → {MINI_PREFIX}")
    print(f"  big   → {BIG_PREFIX}")
    print(f"  out   → {OUT_PREFIX}")

    rng = random.Random(SEED)
    info = json.loads((DATA / "dataset_info.json").read_text())

    summary_rows = []
    for split in SPLITS:
        big = json.load(open(DATA / f"{BIG_PREFIX}_{split}/data.json"))

        if args.source == "mini":
            mini = json.load(open(DATA / f"{MINI_PREFIX}_{split}/data.json"))
            mini_ids = {s["_metadata"]["arxiv_id"] for s in mini}
            mini_by_va = defaultdict(list)
            for s in mini:
                mini_by_va[(group_of(venue(s)), "A" if is_accept(s) else "R")].append(s)

            pool_by_va = defaultdict(list)
            for s in big:
                if s["_metadata"]["arxiv_id"] in mini_ids:
                    continue
                pool_by_va[(group_of(venue(s)), "A" if is_accept(s) else "R")].append(s)

            groups_iter = sorted(set(k[0] for k in mini_by_va.keys()))
        else:  # source == "big"
            big_by_va = defaultdict(list)
            for s in big:
                big_by_va[(group_of(venue(s)), "A" if is_accept(s) else "R")].append(s)
            groups_iter = sorted(set(k[0] for k in big_by_va.keys()))

        out = []
        for v in groups_iter:
            rate = ACCEPT_RATE.get(v)
            if rate is None:
                raise RuntimeError(f"Missing accept rate for group: {v!r}")

            if args.source == "mini":
                n = sum(len(mini_by_va.get((v, lab), [])) for lab in ("A", "R"))
                target_a = round(n * rate)
                target_r = n - target_a

                mini_a = list(mini_by_va.get((v, "A"), []))
                mini_r = list(mini_by_va.get((v, "R"), []))
                pool_a = list(pool_by_va.get((v, "A"), []))
                pool_r = list(pool_by_va.get((v, "R"), []))
                rng.shuffle(mini_a); rng.shuffle(mini_r)
                rng.shuffle(pool_a); rng.shuffle(pool_r)

                avail_r = len(mini_r) + len(pool_r)
                avail_a = len(mini_a) + len(pool_a)

                capped_target_r = min(target_r, avail_r)
                adjusted_target_a = n - capped_target_r
                if adjusted_target_a > avail_a:
                    raise RuntimeError(
                        f"{split}/{v}: not enough samples — n={n} avail_a={avail_a} avail_r={avail_r}"
                    )

                need_more_r = capped_target_r - len(mini_r)
                if need_more_r <= 0:
                    chosen_r = mini_r[:capped_target_r]
                else:
                    chosen_r = mini_r + pool_r[:need_more_r]

                need_more_a = adjusted_target_a - len(mini_a)
                if need_more_a <= 0:
                    chosen_a = mini_a[:adjusted_target_a]
                else:
                    chosen_a = mini_a + pool_a[:need_more_a]

                cell = chosen_a + chosen_r
                rng.shuffle(cell)
                out.extend(cell)

                summary_rows.append({
                    "split": split, "group": v,
                    "n_total": n,
                    "mini_a": len(mini_a),
                    "mini_r": len(mini_r),
                    "pool_r_available": len(pool_r),
                    "target_a": target_a,
                    "target_r": target_r,
                    "got_a": len(chosen_a),
                    "got_r": len(chosen_r),
                    "rate_target": rate,
                    "rate_realized": (len(chosen_a) / n) if n > 0 else 0.0,
                    "capped": capped_target_r != target_r,
                })
            else:  # source == "big": keep all rejects, subsample accepts to hit rate
                big_a = list(big_by_va.get((v, "A"), []))
                big_r = list(big_by_va.get((v, "R"), []))
                rng.shuffle(big_a); rng.shuffle(big_r)

                if len(big_r) == 0:
                    # No rejects in source for this venue (e.g. emnlp/naacl).
                    # Drop this venue entirely — keeping accepts alone defeats the rate target.
                    summary_rows.append({
                        "split": split, "group": v,
                        "n_total": 0,
                        "mini_a": len(big_a),  # reusing slot for "big_a" in this branch
                        "mini_r": 0,
                        "pool_r_available": 0,
                        "target_a": 0,
                        "target_r": 0,
                        "got_a": 0,
                        "got_r": 0,
                        "rate_target": rate,
                        "rate_realized": 0.0,
                        "capped": True,  # will print as "(NO REJECTS, dropped)"
                    })
                    continue

                # Math: target_a / (target_a + len(big_r)) = rate  =>  target_a = R * rate / (1-rate)
                target_a = round(len(big_r) * rate / (1.0 - rate))
                got_a = min(target_a, len(big_a))
                chosen_r = big_r            # keep all
                chosen_a = big_a[:got_a]    # subsample accepts (already shuffled)

                cell = chosen_a + chosen_r
                rng.shuffle(cell)
                out.extend(cell)

                n_realized = got_a + len(chosen_r)
                summary_rows.append({
                    "split": split, "group": v,
                    "n_total": n_realized,
                    "mini_a": len(big_a),       # reusing slot to print "big_a" here
                    "mini_r": len(big_r),       # reusing slot to print "big_r" here
                    "pool_r_available": 0,
                    "target_a": target_a,
                    "target_r": len(chosen_r),
                    "got_a": got_a,
                    "got_r": len(chosen_r),
                    "rate_target": rate,
                    "rate_realized": (got_a / n_realized) if n_realized > 0 else 0.0,
                    "capped": got_a < target_a,  # ran out of accepts
                })

        rng.shuffle(out)

        out_dir = DATA / f"{OUT_PREFIX}_{split}"
        out_path = out_dir / "data.json"
        if out_path.exists():
            raise RuntimeError(
                f"Refusing to overwrite existing dataset at {out_path}. "
                f"Delete it manually if you really intended a re-run."
            )
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(out, f)

        # register in dataset_info.json
        name = f"{OUT_PREFIX}_{split}"
        info[name] = {
            "file_name": f"{name}/data.json",
            "formatting": "sharegpt",
            "columns": {"messages": "conversations"},
            "tags": {
                "role_tag": "from", "content_tag": "value",
                "user_tag": "human", "assistant_tag": "gpt", "system_tag": "system",
            },
        }
        print(f"  wrote {out_dir}/data.json: {len(out)} samples")

    with open(DATA / "dataset_info.json", "w") as f:
        json.dump(info, f, indent=2)
    print(f"\nRegistered 3 datasets in dataset_info.json")

    # --- per-cell summary ---
    print()
    print(f"{'split':<6} {'group':<12} {'n':>5} {'mini_a':>7} {'mini_r':>7} {'pool_r':>7} {'tgt_a':>6} {'tgt_r':>6} {'got_a':>6} {'got_r':>6} {'tgt_rate':>9} {'real_rate':>9} {'flag'}")
    print("-" * 105)
    for r in summary_rows:
        flag = " (CAPPED, no rejects available)" if r["capped"] else ""
        print(f"{r['split']:<6} {r['group']:<12} {r['n_total']:>5} {r['mini_a']:>7} {r['mini_r']:>7} {r['pool_r_available']:>7} "
              f"{r['target_a']:>6} {r['target_r']:>6} {r['got_a']:>6} {r['got_r']:>6} "
              f"{r['rate_target']*100:>8.1f}% {r['rate_realized']*100:>8.1f}%{flag}")


if __name__ == "__main__":
    main()
