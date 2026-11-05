"""paperprep — minimal batch LaTeX/PDF -> ShareGPT pipeline.

This package contains only what's strictly needed to run the `paperprep` CLI
(compile -> MinerU -> normalize -> filter -> export). Higher-level utilities
(openreview ingestion, HF dataset assembly, per-venue balancing, multi-split
export) live in the upstream `simple-cli` / `mergedtree` branches of the same
repository — not here.
"""

from paper_anonymizer.core.filters import FilterMode, filter_content_list

__version__ = "0.1.0"

__all__ = ["FilterMode", "filter_content_list"]
