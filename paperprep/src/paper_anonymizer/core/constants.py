"""Shared regex patterns and constants for content filtering."""

import re

# Section header patterns (support flexible spacing from OCR)
REFERENCES_PATTERN = re.compile(r'^r\s*e\s*f\s*e\s*r\s*e\s*n\s*c\s*e\s*s?$', re.IGNORECASE)
ABSTRACT_PATTERN = re.compile(r'^a\s*b\s*s\s*t\s*r\s*a\s*c\s*t$', re.IGNORECASE)

# Patterns for sections to remove (between body and references)
REPRODUCIBILITY_PATTERN = re.compile(r'repro', re.IGNORECASE)
ACKNOWLEDGMENTS_PATTERN = re.compile(r'acknowledg', re.IGNORECASE)
# NeurIPS Paper Checklist (standard template section, usually appendix).
CHECKLIST_PATTERN = re.compile(r'(?:neurips\s+)?paper\s+checklist', re.IGNORECASE)

# URL detection — removes sentences containing URLs
URL_SENTENCE_PATTERN = re.compile(r'https:\s*//', re.IGNORECASE)

# Author affiliation patterns (page 0 only, for deanonymized mode)
AFFILIATION_PATTERNS = [
    re.compile(r'equal\s+contribution', re.IGNORECASE),
    re.compile(r'correspondence\s+to', re.IGNORECASE),
    re.compile(r'proceedings\s+of\s+the', re.IGNORECASE),
    re.compile(r'international\s+conference\s+on', re.IGNORECASE),
    re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'),  # email
]

# ---------------------------------------------------------------------------
# ShareGPT prompts for LLaMA Factory fine-tuning
# ---------------------------------------------------------------------------
# Verbatim from scripts/create_sharegpt_dataset_for_llama_factory.py —
# these are the exact prompts used in all production ICLR/NeurIPS/TMLR datasets.

SHAREGPT_SYSTEM_PROMPT = (
    "You are an expert academic reviewer tasked with evaluating research papers."
)

SHAREGPT_USER_PROMPT_BINARY = (
    "I am giving you a paper. I want to predict its acceptance outcome at ICLR.\n"
    " - Your answer will either be: \\boxed{Accept} or \\boxed{Reject}\n"
    " - Note: ICLR generally has a ~30% acceptance rate"
)

# Generic prompt for the multi-venue arxiv ShareGPT pipeline (no ICLR-specific
# language). Used by build_sharegpt_per_venue().
SHAREGPT_USER_PROMPT_GENERIC = (
    "I am giving you a paper submitted to a top machine-learning venue. "
    "Predict its acceptance outcome.\n"
    " - Your answer will either be: \\boxed{Accept} or \\boxed{Reject}\n"
    " - Note: typical top-tier ML venues have ~25-30% acceptance rates"
)
