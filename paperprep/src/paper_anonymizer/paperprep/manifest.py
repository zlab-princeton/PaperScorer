"""Input manifest parsing for paperprep.

Manifest is JSONL — one paper per line. Each row is one of:

    {"id": "paper001", "type": "latex_dir", "path": "/data/papers/paper001/"}
    {"id": "paper002", "type": "latex_dir", "path": "/data/papers/paper002/", "main_tex": "main.tex"}
    {"id": "paper003", "type": "pdf", "path": "/data/papers/paper003.pdf"}

`main_tex` is optional for latex_dir; auto-detected via `00README.json` then
`\\documentclass` heuristic if not given. `id` must be unique.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal


PaperType = Literal["latex_dir", "pdf"]


@dataclass
class PaperEntry:
    id: str
    type: PaperType
    path: Path
    main_tex: str | None = None  # only used when type == "latex_dir"
    extra: dict = field(default_factory=dict)  # passthrough for caller-defined fields


def _validate_row(row: dict, line_no: int) -> PaperEntry:
    missing = [k for k in ("id", "type", "path") if k not in row]
    if missing:
        raise ValueError(f"manifest line {line_no}: missing fields {missing}")

    pt = row["type"]
    if pt not in ("latex_dir", "pdf"):
        raise ValueError(f"manifest line {line_no}: type must be 'latex_dir' or 'pdf', got {pt!r}")

    path = Path(row["path"]).expanduser().resolve()
    if pt == "latex_dir":
        if not path.is_dir():
            raise ValueError(f"manifest line {line_no}: latex_dir path does not exist or is not a directory: {path}")
    else:
        if not path.is_file():
            raise ValueError(f"manifest line {line_no}: pdf path does not exist or is not a file: {path}")

    main_tex = row.get("main_tex")
    extra = {k: v for k, v in row.items() if k not in ("id", "type", "path", "main_tex")}
    return PaperEntry(id=str(row["id"]), type=pt, path=path, main_tex=main_tex, extra=extra)


def load_manifest(manifest_path: Path) -> list[PaperEntry]:
    """Parse a manifest JSONL file. Validates every row eagerly."""
    rows: list[PaperEntry] = []
    seen_ids: set[str] = set()
    with open(manifest_path, "r") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"manifest line {line_no}: invalid JSON: {e}") from e
            entry = _validate_row(row, line_no)
            if entry.id in seen_ids:
                raise ValueError(f"manifest line {line_no}: duplicate id {entry.id!r}")
            seen_ids.add(entry.id)
            rows.append(entry)
    if not rows:
        raise ValueError(f"manifest is empty: {manifest_path}")
    return rows


def build_manifest_from_dir(input_dir: Path, mode: PaperType) -> list[PaperEntry]:
    """Auto-build a manifest by scanning a directory.

    For mode="latex_dir": treats every immediate sub-dir as a paper (id = subdir name).
    For mode="pdf": every *.pdf in the dir (id = stem of filename).
    """
    input_dir = input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise ValueError(f"input dir does not exist: {input_dir}")

    rows: list[PaperEntry] = []
    if mode == "latex_dir":
        for sub in sorted(input_dir.iterdir()):
            if sub.is_dir():
                rows.append(PaperEntry(id=sub.name, type="latex_dir", path=sub))
    elif mode == "pdf":
        for pdf in sorted(input_dir.glob("*.pdf")):
            rows.append(PaperEntry(id=pdf.stem, type="pdf", path=pdf))
    else:
        raise ValueError(f"mode must be 'latex_dir' or 'pdf', got {mode!r}")

    if not rows:
        raise ValueError(f"no {mode} entries found in {input_dir}")
    return rows


def write_manifest(entries: Iterable[PaperEntry], path: Path) -> None:
    """Persist a manifest in canonical JSONL form."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for e in entries:
            row = {"id": e.id, "type": e.type, "path": str(e.path)}
            if e.main_tex:
                row["main_tex"] = e.main_tex
            row.update(e.extra)
            f.write(json.dumps(row) + "\n")
