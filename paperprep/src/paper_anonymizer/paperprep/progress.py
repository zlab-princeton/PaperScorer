"""tqdm progress + structured logging for paperprep.

Two flavors:

- `paper_iter()`: tqdm-wrapped iterator for per-paper loops inside a worker /
  chunk function. Writes to the worker's log file (visible by tail-ing
  submitit logs from the login node).

- `submitit_progress()`: driver-side bar that polls a list of submitit jobs
  and updates as each transitions to a terminal state (DONE / FAILED).
"""
from __future__ import annotations

import time
from typing import Iterable, Iterator, Sequence, TypeVar

from tqdm import tqdm

T = TypeVar("T")


def paper_iter(items: Sequence[T], stage: str, total: int | None = None) -> Iterator[T]:
    """tqdm-wrapped iterator. Use inside a worker / chunk function."""
    return tqdm(
        items,
        desc=f"[{stage}]",
        total=total if total is not None else len(items),
        unit="paper",
        smoothing=0.1,
        mininterval=0.5,
    )


def submitit_progress(jobs: list, stage: str, poll_interval: float = 2.0) -> dict:
    """Driver-side bar over a list of submitit Jobs.

    Polls `job.state` until each is in a terminal state. Returns a dict mapping
    {job_id: final_state}. Caller is responsible for fetching results separately.
    """
    final_states: dict[str, str] = {}
    pending = list(jobs)
    bar = tqdm(total=len(pending), desc=f"[{stage} jobs]", unit="job", smoothing=0.1)
    while pending:
        still_pending = []
        for j in pending:
            try:
                state = j.state  # property; may hit slurm
            except Exception:
                state = "UNKNOWN"
            # Terminal states submitit defines: COMPLETED / FAILED / CANCELLED / TIMEOUT / OUT_OF_MEMORY
            if state in ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY"):
                final_states[str(j.job_id)] = state
                bar.update(1)
            else:
                still_pending.append(j)
        pending = still_pending
        if pending:
            time.sleep(poll_interval)
    bar.close()
    return final_states
