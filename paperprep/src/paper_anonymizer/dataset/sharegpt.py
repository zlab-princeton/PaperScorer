"""ShareGPT conversation builders for paperprep.

Only the low-level `create_sharegpt_conversation` + `save_sharegpt_dataset`
helpers are kept here. The higher-level HF-dataset-aware builders
(`build_sharegpt_from_hf_dataset`, `balance_per_year`, `venue_stratified_split`,
the per-row `_build_arxiv_*_entry` adapters, etc.) live in the upstream
`simple-cli`/`mergedtree` branches — they pull in the HuggingFace `datasets`
library which paperprep doesn't need.

For inference-only batch processing (paperprep's use case), pass `label=None`
to omit the gpt turn entirely.
"""

import json
from pathlib import Path

from paper_anonymizer.core.constants import (
    SHAREGPT_SYSTEM_PROMPT,
    SHAREGPT_USER_PROMPT_BINARY,
)


def create_sharegpt_conversation(
    paper_text: str,
    label: str | None,
    system_prompt: str = SHAREGPT_SYSTEM_PROMPT,
    task_prompt: str = SHAREGPT_USER_PROMPT_BINARY,
    metadata: dict | None = None,
    images: list[str] | None = None,
) -> dict:
    """Create a single ShareGPT-format conversation.

    Args:
        paper_text: The paper content (markdown or <image> placeholders).
        label: 'Accept' or 'Reject'. Pass ``None`` for inference-only entries
            (no gpt turn emitted) — used by paperprep when batch-processing
            unlabeled user-supplied papers.
        system_prompt: System message.
        task_prompt: User prompt template.
        metadata: Optional metadata to attach.
        images: Optional list of image paths (for vision).

    Returns:
        ShareGPT conversation dict.
    """
    human_content = f"{task_prompt}\n\n{paper_text}"

    conversations = [
        {"from": "system", "value": system_prompt},
        {"from": "human", "value": human_content},
    ]
    if label is not None:
        conversations.append({"from": "gpt", "value": f"Outcome: \\boxed{{{label}}}"})

    entry = {"conversations": conversations}

    if metadata:
        entry["_metadata"] = metadata
    if images:
        entry["images"] = images

    return entry


def save_sharegpt_dataset(entries: list[dict], output_path: Path) -> None:
    """Save a list of ShareGPT entries to a JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(entries, f, indent=2)
    print(f"Saved {len(entries)} entries to {output_path}")
