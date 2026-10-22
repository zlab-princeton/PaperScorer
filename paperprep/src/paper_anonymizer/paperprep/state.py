"""Per-paper, per-stage state tracking for paperprep.

Append-only JSONL log at `<output_dir>/state.jsonl`. Each row records one
(stage, paper_id) outcome. Drives `--resume`: a paper that already has a
"status=ok" row for a stage will be skipped on the next run.

Schema:
    {"id": ..., "stage": ..., "status": "ok"|"failed"|"skipped",
     "error": "<msg>" | null, "output_path": "<path>" | null,
     "started_at": <iso8601>, "ended_at": <iso8601>}
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Literal

Status = Literal["ok", "failed", "skipped"]


@dataclass
class StageResult:
    id: str
    stage: str
    status: Status
    error: str | None = None
    output_path: str | None = None
    started_at: float = field(default_factory=time.time)
    ended_at: float = field(default_factory=time.time)
    extra: dict = field(default_factory=dict)


class StateLog:
    """Append-only JSONL of per-(paper, stage) results."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # cache of (id, stage) -> latest status
        self._latest: dict[tuple[str, str], str] = {}
        if self.path.exists():
            with open(self.path) as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                        self._latest[(row["id"], row["stage"])] = row["status"]
                    except (json.JSONDecodeError, KeyError):
                        continue

    def append(self, result: StageResult) -> None:
        row = asdict(result)
        with open(self.path, "a") as f:
            f.write(json.dumps(row) + "\n")
        self._latest[(result.id, result.stage)] = result.status

    def append_many(self, results: Iterable[StageResult]) -> None:
        for r in results:
            self.append(r)

    def already_ok(self, paper_id: str, stage: str) -> bool:
        return self._latest.get((paper_id, stage)) == "ok"

    def filter_unprocessed(self, ids: list[str], stage: str) -> list[str]:
        """Return ids that don't already have status=ok for this stage."""
        return [i for i in ids if not self.already_ok(i, stage)]

    def summarize(self) -> dict[str, dict[Status, int]]:
        """Return {stage: {status: count}} across the entire log."""
        summary: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        if not self.path.exists():
            return dict(summary)
        with open(self.path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    summary[row["stage"]][row["status"]] += 1
                except (json.JSONDecodeError, KeyError):
                    continue
        return {k: dict(v) for k, v in summary.items()}
