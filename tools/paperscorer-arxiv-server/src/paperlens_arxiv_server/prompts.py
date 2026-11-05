"""Prompt blocks that match the released PaperLens HF model cards exactly.

The reranker MUST send the same SYSTEM + USER preamble its model card was
trained against. We score the logprob of `\\boxed{Accept}` vs `\\boxed{Reject}`
at the boxed-token position, mirroring how training labels were written.
"""
from __future__ import annotations


SHAREGPT_SYSTEM_PROMPT = (
    "You are an expert academic reviewer tasked with evaluating research papers."
)

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


def prompt_for(domain: str) -> str:
    """Return the user prompt for the named training domain."""
    d = domain.lower()
    if d == "arxiv":
        return PROMPT_ARXIV
    if d == "iclr":
        return PROMPT_ICLR
    raise ValueError(f"unknown domain {domain!r} (expected 'arxiv' or 'iclr')")


def build_user_turn(domain: str, title: str, body: str, n_images: int = 0,
                    abstract: str = "") -> str:
    """Compose the human-turn value the way the HF dataset rows are formatted.

    Text: prompt + paper body (which already starts with `# {title}`).
    Vision: prompt + `# {title}\n\n## Abstract\n{abstract}\n\n<image> <image> ...`.
    """
    prompt = prompt_for(domain)
    if n_images > 0:
        # Vision-style
        parts = [prompt, "", f"# {title.strip()}"]
        if abstract.strip():
            parts.append(f"## Abstract\n{abstract.strip()}")
        parts.append(" ".join(["<image>"] * n_images))
        return "\n\n".join(parts)
    # Text-style -- body should already start with `# {title}\n\n...`
    return f"{prompt}\n\n{body.strip()}"
