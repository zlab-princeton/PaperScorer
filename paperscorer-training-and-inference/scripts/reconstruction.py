"""Reconstruct LlamaFactory-formatted local datasets from the PaperLens HF release.

Hub-native by default: pulls the manifest + parquet shards from
``skonan/PaperScorer-Text`` and ``skonan/PaperScorer-Vision`` on the HF Hub.
Pass ``--local_dir`` only when re-running against an offline mirror.

Usage:

    # Reconstruct the canonical 4-cell eval datasets (hub-native)
    python scripts/reconstruction.py --dataset_keys \\
        arxiv_50_50_21k_text_wmetadata_filtered24480_y24up_test \\
        iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_y25up_test

    # Rebuild everything (large; writes the full vision image tree)
    python scripts/reconstruction.py --all

    # Validate data.json only, skip the ~300 GB image tree
    python scripts/reconstruction.py --all --dry-run

Round-trip is byte-identical for conv[1] + ``_metadata`` + image bytes;
filtering is columnar (only the parquet ``references`` column is scanned
to build the per-key row-index map) so the image bytes are never decoded
during the scan.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# These prompts are constant per family. The HF dataset stores `content` with
# whatever whitespace originally separated the prompt from the paper body
# (legacy ICLR text rows have an extra blank line), so reconstruction is just
# `prompt + content` -- no separator guesswork.
SHAREGPT_SYSTEM_PROMPT = "You are an expert academic reviewer tasked with evaluating research papers."

PROMPT_ARXIV = (
    "I am giving you a paper submitted to a top machine-learning venue. "
    "Predict its acceptance outcome.\n"
    " - Your answer will either be: \\boxed{Accept} or \\boxed{Reject}\n"
    " - Note: typical top-tier ML venues have ~25-30% acceptance rates"
)
PROMPT_ICLR = (
    "I am giving you a paper. I want to predict its acceptance outcome at ICLR.\n"
    " - Your answer will either be: \\boxed{Accept} or \\boxed{Reject}\n"
    " - Note: ICLR generally has a ~30% acceptance rate"
)

# Default HF repo names. Override via --hf_text_repo / --hf_vision_repo.
DEFAULT_TEXT_REPO = "skonan/PaperScorer-Text"
DEFAULT_VISION_REPO = "skonan/PaperScorer-Vision"

# Columns we need per matched row (everything except the heavy `images` blob).
LIGHT_COLS = ["paper_id", "content", "metadata", "label"]


def prompt_for_subset(subset: str) -> str:
    if subset == "arxiv":
        return PROMPT_ARXIV
    if subset == "iclr":
        return PROMPT_ICLR
    raise ValueError(f"unknown paper subset: {subset!r}")


def modality_of(key: str) -> str:
    if "_text_" in key or key.startswith("combined_arxiv_iclr_42k_text"):
        return "text"
    if "_vision_" in key or key.startswith("combined_arxiv_iclr_42k_vision"):
        return "vision"
    raise ValueError(f"cannot infer modality from key: {key!r}")


def gpt_turn_for(label: str) -> str:
    return f"Outcome: \\boxed{{{label}}}"


def _build_ref_index(references_col) -> dict:
    """references_col: iterable of `list[[name, split], ...]`.
    Returns {(name, split): [row_idx, ...]} preserving row order.
    """
    index: dict = {}
    for i, refs in enumerate(references_col):
        for r in (refs or []):
            index.setdefault((r[0], r[1]), []).append(i)
    return index


class _LocalParquetReader:
    """Lazy reader over one (modality, subset) backed by local parquet shards.

    The (name, split) -> indices map is built from the `references` column only;
    image bytes are read (decode-free) solely for the matched rows we ask for.
    """

    def __init__(self, parquet_paths: list[Path]):
        import pyarrow.dataset as pds
        self._pa = pa = __import__("pyarrow")
        paths = [str(p) for p in parquet_paths]
        base = pds.dataset(paths, format="parquet")
        # Promote string -> large_string so reading a >2GB content column (full
        # paper bodies in text/arxiv) doesn't overflow pyarrow's int32 offsets.
        def _promote(t):
            if t == pa.string():
                return pa.large_string()
            if isinstance(t, pa.ListType):
                return pa.list_(_promote(t.value_type))
            return t
        schema = pa.schema([pa.field(f.name, _promote(f.type)) for f in base.schema])
        self._dset = pds.dataset(paths, format="parquet", schema=schema)
        names = set(self._dset.schema.names)
        self.has_images = "images" in names
        refs = self._dset.to_table(columns=["references"]).column("references").to_pylist()
        self.index = _build_ref_index(refs)
        self._light = None  # cached light-column table
        # Row-group map in global (dataset/to_table) row order, so vision image
        # bytes can be streamed one ~390MB row group at a time. Footers only.
        self._rg_index: list[tuple[int, str, int]] = []   # (global_start, path, rg_idx)
        self._rg_starts: list[int] = []
        if self.has_images:
            import pyarrow.parquet as pq
            g = 0
            for p in paths:
                md = pq.ParquetFile(p).metadata
                for rg in range(md.num_row_groups):
                    self._rg_index.append((g, p, rg))
                    self._rg_starts.append(g)
                    g += md.row_group(rg).num_rows

    def _light_table(self):
        if self._light is None:
            cols = [c for c in LIGHT_COLS if c in self._dset.schema.names]
            self._light = self._dset.to_table(columns=cols)
        return self._light

    def light_rows(self, indices: list[int]) -> list[dict]:
        idx = self._pa.array(indices, type=self._pa.int64())
        return self._light_table().take(idx).to_pylist()

    def write_images(self, indices: list[int], pids: list[str],
                     data_root: Path, subset: str) -> None:
        """Stream matched rows' images one row group at a time (~390MB peak) and
        write each paper's PNGs. Each needed row group is read exactly once.

        `dataset.take([scattered indices])` would pull every touched row group
        (100 rows / ~390MB each) into one Arrow table -- a sparse match set spans
        ~all groups, so that OOMs. Grouping by row group bounds peak RAM instead.
        """
        import bisect
        import pyarrow.parquet as pq
        from collections import defaultdict
        by_rg: dict[int, list[tuple[int, str]]] = defaultdict(list)
        for gi, pid in zip(indices, pids):
            ri = bisect.bisect_right(self._rg_starts, gi) - 1
            by_rg[ri].append((gi - self._rg_starts[ri], pid))
        for ri in sorted(by_rg):
            _, path, rg_idx = self._rg_index[ri]
            col = pq.ParquetFile(path).read_row_group(rg_idx, columns=["images"]).column("images").to_pylist()
            for local, pid in by_rg[ri]:
                d = data_root / f"images_{subset}" / pid
                d.mkdir(parents=True, exist_ok=True)
                for i, im in enumerate(col[local] or []):
                    _write_png(im, d / f"page_{i + 1}.png")
            del col


class _HubDatasetReader:
    """Lazy reader over an in-memory HF Dataset (from the hub or save_to_disk).

    Index scan uses select_columns(['references']) (no image decode); matched
    rows are pulled via .select(indices) with the image column cast to
    decode=False so we get raw bytes, never PIL-decoded pixels.
    """

    def __init__(self, ds):
        self._ds = ds
        self.has_images = "images" in ds.features
        refs = ds.select_columns(["references"])["references"]
        self.index = _build_ref_index(refs)

    def _selected_decodefree(self, indices: list[int]):
        sel = self._ds.select(indices)
        if self.has_images:
            from datasets import Image
            feat = sel.features["images"]
            inner = Image(decode=False)
            wrapper = type(feat)(inner) if hasattr(feat, "feature") else [inner]
            sel = sel.cast_column("images", wrapper)
        return sel

    def light_rows(self, indices: list[int]) -> list[dict]:
        sel = self._ds.select_columns(
            [c for c in LIGHT_COLS if c in self._ds.features]
        ).select(indices)
        return sel.to_list()

    def write_images(self, indices: list[int], pids: list[str],
                     data_root: Path, subset: str, chunk: int = 32) -> None:
        """Write matched rows' PNGs in bounded chunks (decode-free), so peak RAM
        stays ~`chunk` papers rather than the whole match set."""
        for s in range(0, len(indices), chunk):
            sel = self._selected_decodefree(indices[s:s + chunk])
            for row, pid in zip(sel, pids[s:s + chunk]):
                d = data_root / f"images_{subset}" / pid
                d.mkdir(parents=True, exist_ok=True)
                for i, im in enumerate(row["images"] or []):
                    _write_png(im, d / f"page_{i + 1}.png")


def _write_png(blob, out_path: Path) -> None:
    """blob is {'bytes':..,'path':..} (decode-free), raw bytes, or a PIL image."""
    if isinstance(blob, dict):
        out_path.write_bytes(blob["bytes"])
    elif isinstance(blob, (bytes, bytearray)):
        out_path.write_bytes(blob)
    elif hasattr(blob, "save"):
        blob.save(out_path, format="PNG")
    else:
        raise TypeError(f"cannot write image of type {type(blob)!r}")


def build_text_row(row: dict, subset: str, emit_label: bool = False) -> dict:
    prompt = prompt_for_subset(subset)
    human_value = f"{prompt}{row['content']}"
    label = row.get("label") or ""
    conversations = [
        {"from": "system", "value": SHAREGPT_SYSTEM_PROMPT},
        {"from": "human", "value": human_value},
    ]
    if label:
        conversations.append({"from": "gpt", "value": gpt_turn_for(label)})
    out = {"conversations": conversations, "_metadata": json.loads(row["metadata"])}
    if emit_label:
        # accept_reject_label: 1=Accept, 0=Reject (binary-classification column;
        # LlamaFactory reads it when the key's columns mapping references it).
        out["accept_reject_label"] = 1 if label == "Accept" else 0
    return out


def build_vision_row(row: dict, subset: str, data_root: Path, n_pages: int,
                     emit_label: bool = False) -> dict:
    """Build the sharegpt row with image PATHS only. The PNG bytes are written
    separately by the reader's streaming write_images (memory-bounded)."""
    out = build_text_row(row, subset, emit_label)
    pid = row["paper_id"]
    rel_dir = data_root / f"images_{subset}" / pid
    out["images"] = [str((rel_dir / f"page_{i + 1}.png").relative_to(data_root.parent))
                     for i in range(n_pages)]
    return out


def reconstruct_key(key: str, key_meta: dict, dataset_info: dict, readers: dict,
                    data_root: Path, dry_run: bool = False, limit: int = 0) -> int:
    """Materialize ``<data_root>/<key>/data.json`` (+ images for vision) from the
    HF rows whose `references` contain this key's friendly (name, split).
    Returns the number of rows written. limit>0 caps rows (for sampling).
    """
    modality = modality_of(key)
    meta = key_meta.get(key)
    if meta is None:
        print(f"  [SKIP] {key}: not in release manifest", file=sys.stderr, flush=True)
        return 0
    want = meta["ref"]
    columns = meta["columns"] or {"messages": "conversations"}
    # The key carries the accept_reject_label column only when its dataset_info
    # columns mapping references it (train keys do; test keys + *_50pct don't).
    emit_label = "accept_reject_label" in columns
    mod_readers = readers[modality]

    rows_out: list[dict] = []
    for subset in ("arxiv", "iclr"):
        rd = mod_readers[subset]
        indices = rd.index.get(want, [])
        if limit:
            indices = indices[: max(0, limit - len(rows_out))]
        if not indices:
            continue
        light = rd.light_rows(indices)
        if modality == "vision":
            # Page count is free from the <image> placeholders baked into
            # content -- no image read needed to count or to emit paths. The PNG
            # bytes are then streamed to disk row-group-by-row-group, so peak RAM
            # is one ~390MB row group regardless of key size (a full vision train
            # key is ~70K papers x ~7 pages).
            pids = [row["paper_id"] for row in light]
            for row in light:
                n_pages = row["content"].count("<image>")
                rows_out.append(build_vision_row(row, subset, data_root, n_pages, emit_label))
            if not dry_run:
                rd.write_images(indices, pids, data_root, subset)
        else:
            for row in light:
                rows_out.append(build_text_row(row, subset, emit_label))
        if limit and len(rows_out) >= limit:
            break

    if not rows_out:
        print(f"  [SKIP] {key}: no matching papers", file=sys.stderr, flush=True)
        return 0

    out_dir = data_root / key
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "data.json"
    out_path.write_text(json.dumps(rows_out, indent=2))

    # Reproduce the original dataset_info entry verbatim from the manifest
    # (columns mapping + file_name) so LlamaFactory loads it identically.
    dataset_info[key] = {
        "file_name": meta["file_name"],
        "formatting": "sharegpt",
        "columns": columns,
        "tags": {
            "role_tag": "from",
            "content_tag": "value",
            "user_tag": "human",
            "assistant_tag": "gpt",
            "system_tag": "system",
        },
    }
    print(f"  [OK]   {key}: {len(rows_out)} rows -> {out_path}", flush=True)
    return len(rows_out)


def _build_readers(args, modalities):
    """Return {'text': {'arxiv': reader, 'iclr': reader}, 'vision': {...}}.

    `modalities` is the set of {'text','vision'} actually needed by the requested
    keys -- we lazily load only those (so reconstructing a text-only key doesn't
    download the 472GB vision dataset from the hub).
    """
    print(f"[reconstruction] indexing HF references for {sorted(modalities)} "
          "(columnar, no image decode) ...", flush=True)
    readers: dict = {"text": {}, "vision": {}}
    if args.local_dir:
        for modality in modalities:
            for sub in ("arxiv", "iclr"):
                d = args.local_dir / modality / sub
                parts = sorted(d.glob("*.parquet"))
                if parts:
                    readers[modality][sub] = _LocalParquetReader(parts)
                else:
                    from datasets import load_from_disk
                    readers[modality][sub] = _HubDatasetReader(load_from_disk(str(d)))
    else:
        # Hub config names use a friendlier 'openreview-iclr' label; the internal
        # subset key 'iclr' is kept for prompt selection / image dir naming.
        HUB_NAME = {"arxiv": "arxiv", "iclr": "openreview-iclr"}
        from datasets import load_dataset
        for modality in modalities:
            repo = args.hf_text_repo if modality == "text" else args.hf_vision_repo
            for sub in ("arxiv", "iclr"):
                readers[modality][sub] = _HubDatasetReader(
                    load_dataset(repo, name=HUB_NAME[sub], split="papers"))
    return readers


def load_manifest(args) -> dict:
    """Load manifest.json: from --local_dir if given, otherwise download it from
    the configured HF text-dataset repo (it's the same manifest text + vision share)."""
    if args.local_dir:
        return json.loads((args.local_dir / "manifest.json").read_text())
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(repo_id=args.hf_text_repo, filename="manifest.json",
                        repo_type="dataset")
    return json.loads(Path(p).read_text())


def list_all_publishable_keys(manifest: dict) -> list[str]:
    return sorted(manifest["keys"].keys())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset_keys", nargs="*",
                    help="dataset_info.json keys to reconstruct")
    ap.add_argument("--all", action="store_true",
                    help="Reconstruct every key in the HF release manifest")
    ap.add_argument("--local_dir", type=Path, default=None,
                    help="Path to a locally-built HF release dir (skip hub fetch)")
    ap.add_argument("--hf_text_repo", default=DEFAULT_TEXT_REPO)
    ap.add_argument("--hf_vision_repo", default=DEFAULT_VISION_REPO)
    ap.add_argument("--data_root", type=Path, default=Path("data"),
                    help="Local data/ root to materialize into")
    ap.add_argument("--dataset_info", type=Path, default=None,
                    help="Path to dataset_info.json (default: data_root/dataset_info.json)")
    ap.add_argument("--dry-run", dest="dry_run", action="store_true",
                    help="vision: emit image paths in data.json but skip writing the PNGs")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap rows reconstructed per key (0 = all); for quick sampling")
    args = ap.parse_args()

    if not args.dataset_keys and not args.all:
        ap.error("either --dataset_keys ... or --all is required")

    try:
        import pyarrow  # noqa: F401
        import datasets  # noqa: F401
    except ImportError as e:
        print(f"ERROR: need `datasets` + `pyarrow` + Pillow: {e}", file=sys.stderr)
        return 2

    # Per-key reconstruction metadata: the friendly (name, split) the rows
    # reference, plus the original dataset_info columns mapping + file_name.
    # Manifest comes from --local_dir or the hub (hf_hub_download).
    manifest = load_manifest(args)
    key_meta = {
        k: {
            "ref": (v["release_name"], v["release_split"]),
            "columns": v.get("columns", {}),
            "file_name": v.get("file_name", f"{k}/data.json"),
        }
        for k, v in manifest["keys"].items()
    }

    keys = args.dataset_keys or list_all_publishable_keys(manifest)
    modalities = {modality_of(k) for k in keys}
    readers = _build_readers(args, modalities)
    tags = []
    if args.dry_run:
        tags.append("dry-run")
    if args.limit:
        tags.append(f"limit={args.limit}")
    print(f"[reconstruction] {len(keys)} key(s) to materialize -> {args.data_root}"
          + (f"  ({', '.join(tags)})" if tags else ""), flush=True)

    args.data_root.mkdir(parents=True, exist_ok=True)
    info_path = args.dataset_info or (args.data_root / "dataset_info.json")
    dataset_info = json.loads(info_path.read_text()) if info_path.exists() else {}

    total = 0
    for key in keys:
        try:
            total += reconstruct_key(key, key_meta, dataset_info, readers,
                                     args.data_root, dry_run=args.dry_run, limit=args.limit)
        except Exception as e:
            print(f"  [FAIL] {key}: {e}", file=sys.stderr, flush=True)
        # Write dataset_info incrementally so a partial run is still usable.
        info_path.write_text(json.dumps(dataset_info, indent=2))

    print(f"[reconstruction] done. {total} total rows. dataset_info.json -> {info_path}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
