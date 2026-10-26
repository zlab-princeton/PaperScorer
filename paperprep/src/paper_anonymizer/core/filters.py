"""Unified content list filtering with two modes: ANONYMOUS and DEANONYMIZED.

Both modes always remove references. The difference is what else gets removed.

ANONYMOUS (paper is already anonymous):
    - Keeps title, authors, abstract in natural document order
    - Removes: references, aside_text, type="header"
    - Keeps: ack, repro, appendix, footnotes, page numbers

DEANONYMIZED (paper has real author names):
    - Strips title, authors, abstract (via bbox indent analysis)
    - Removes: references, ack, repro, aside_text, type="header", affiliations, footnotes
    - Keeps: page numbers

Ported from:
    - lib/normalize_mineru.py:filter_content_list() → DEANONYMIZED
    - scripts/generate_original_md_minus_refs.py:filter_content_list_original_minus_refs() → ANONYMOUS
"""

import copy
from enum import Enum

from paper_anonymizer.core.constants import (
    REFERENCES_PATTERN,
    ABSTRACT_PATTERN,
    REPRODUCIBILITY_PATTERN,
    ACKNOWLEDGMENTS_PATTERN,
    URL_SENTENCE_PATTERN,
    AFFILIATION_PATTERNS,
)


class FilterMode(Enum):
    ANONYMOUS = "anonymous"
    DEANONYMIZED = "deanonymized"


def _is_section_header(item: dict) -> bool:
    return item.get('type') == 'text' and item.get('text_level', 0) > 0


def _is_abstract_header(item: dict) -> bool:
    if not _is_section_header(item):
        return False
    return bool(ABSTRACT_PATTERN.match(item.get('text', '').strip()))


def _is_references_header(item: dict) -> bool:
    if not _is_section_header(item):
        return False
    return bool(REFERENCES_PATTERN.match(item.get('text', '').strip()))


def _has_url(text: str) -> bool:
    return bool(URL_SENTENCE_PATTERN.search(text))


def _is_affiliation_text(text: str) -> bool:
    if not text:
        return False
    return any(p.search(text) for p in AFFILIATION_PATTERNS)


def filter_content_list(
    content_list: list[dict],
    mode: FilterMode,
    abstract_text: str | None = None,
) -> tuple[list[dict], dict]:
    """Filter a MinerU content list.

    Args:
        content_list: Raw MinerU content_list items.
        mode: ANONYMOUS (keep title/abstract) or DEANONYMIZED (strip identity).
        abstract_text: For DEANONYMIZED mode, the abstract text to anonymize
                       and prepend as a markdown prefix (returned in metadata).

    Returns:
        (filtered_list, metadata_dict)
    """
    if mode == FilterMode.ANONYMOUS:
        return _filter_anonymous(content_list)
    else:
        return _filter_deanonymized(content_list, abstract_text)


# ---------------------------------------------------------------------------
# ANONYMOUS mode (Level 2: original-minus-refs)
# ---------------------------------------------------------------------------

def _filter_anonymous(content_list: list[dict]) -> tuple[list[dict], dict]:
    """Keep title/abstract, remove only refs + asides + headers."""
    metadata = {
        'mode': 'anonymous',
        'cutoff_idx': -1,
        'items_count': 0,
        'asides_removed': 0,
        'headers_removed': 0,
    }

    if not content_list:
        return [], metadata

    # Find REFERENCES section cutoff
    cutoff_idx = len(content_list)
    for i, item in enumerate(content_list):
        if item.get('text_level') == 1:
            text = item.get('text', '').strip()
            if REFERENCES_PATTERN.search(text):
                cutoff_idx = i
                break

    metadata['cutoff_idx'] = cutoff_idx

    filtered = []
    for i, item in enumerate(content_list):
        if i >= cutoff_idx:
            break

        item_type = item.get('type', '')

        if item_type == 'header':
            metadata['headers_removed'] += 1
            continue

        if item_type == 'aside_text':
            metadata['asides_removed'] += 1
            continue

        item_copy = copy.deepcopy(item)

        # Uppercase H1 headers for consistency
        if item_copy.get('text_level') == 1:
            item_copy['text'] = item_copy.get('text', '').upper()

        filtered.append(item_copy)

    metadata['items_count'] = len(filtered)
    return filtered, metadata


# ---------------------------------------------------------------------------
# DEANONYMIZED mode (Level 1: clean)
# ---------------------------------------------------------------------------

def _filter_deanonymized(
    content_list: list[dict],
    abstract_text: str | None = None,
) -> tuple[list[dict], dict]:
    """Strip title/authors/abstract/ack/repro, remove refs."""
    metadata = {
        'mode': 'deanonymized',
        'abstract_idx': -1,
        'intro_idx': -1,
        'references_idx': -1,
        'references_end_idx': -1,
        'page_end': -1,
        'removed_before_intro': [],
        'removed_footnotes': [],
        'removed_after_refs': [],
        'removed_reproducibility': [],
        'removed_acknowledgments': [],
        'removed_aside_text': [],
        'removed_affiliations': [],
        'headers_to_whiteout': [],
        'has_abstract': False,
        'has_references': False,
        'markdown_prefix': '',
    }

    if not content_list:
        return [], metadata

    # --- Find ABSTRACT header ---
    abstract_idx = -1
    for i, item in enumerate(content_list):
        if _is_abstract_header(item):
            abstract_idx = i
            metadata['abstract_idx'] = i
            metadata['has_abstract'] = True
            break

    if abstract_idx < 0:
        for i, item in enumerate(content_list):
            if item.get('text_level') == 1:
                abstract_idx = i
                metadata['abstract_idx'] = i
                break

    if abstract_idx < 0:
        return [], metadata

    # --- Find intro using bbox indent analysis ---
    intro_idx = abstract_idx + 1

    if abstract_idx + 1 < len(content_list):
        abstract_text_item = content_list[abstract_idx + 1]
        abstract_bbox = abstract_text_item.get('bbox', [])

        if abstract_bbox:
            abstract_left_x = abstract_bbox[0]
            threshold = 30

            for i in range(abstract_idx + 2, len(content_list)):
                item = content_list[i]
                bbox = item.get('bbox', [])

                if bbox:
                    left_x = bbox[0]
                    if abstract_left_x - left_x > threshold:
                        intro_idx = i
                        break

                if item.get('text_level') == 1:
                    intro_idx = i
                    break
        else:
            intro_idx = abstract_idx + 2 if abstract_idx + 2 < len(content_list) else abstract_idx + 1

    metadata['intro_idx'] = intro_idx

    # --- Find REFERENCES header ---
    references_idx = -1
    for i, item in enumerate(content_list):
        if _is_references_header(item):
            references_idx = i
            metadata['references_idx'] = i
            metadata['has_references'] = True
            break

    references_end_idx = len(content_list)
    if references_idx >= 0:
        for i in range(references_idx + 1, len(content_list)):
            if content_list[i].get('text_level', 0) > 0:
                references_end_idx = i
                break
        metadata['references_end_idx'] = references_end_idx

        if references_end_idx > references_idx + 1:
            metadata['page_end'] = content_list[references_end_idx - 1].get('page_idx', 0)
        else:
            metadata['page_end'] = content_list[references_idx].get('page_idx', 0)
    else:
        metadata['page_end'] = max(item.get('page_idx', 0) for item in content_list)

    # --- Find ack/repro sections ---
    reproducibility_start = reproducibility_end = -1
    acknowledgments_start = acknowledgments_end = -1
    search_end = references_idx if references_idx > 0 else len(content_list)

    for i in range(intro_idx, search_end):
        item = content_list[i]
        if item.get('text_level') == 1:
            text = item.get('text', '').strip()
            if REPRODUCIBILITY_PATTERN.search(text):
                reproducibility_start = i
                for j in range(i + 1, search_end):
                    if content_list[j].get('text_level') == 1:
                        reproducibility_end = j
                        break
                else:
                    reproducibility_end = search_end
            elif ACKNOWLEDGMENTS_PATTERN.search(text):
                acknowledgments_start = i
                for j in range(i + 1, search_end):
                    if content_list[j].get('text_level') == 1:
                        acknowledgments_end = j
                        break
                else:
                    acknowledgments_end = search_end

    # --- Build filtered list ---
    filtered = []

    for i, item in enumerate(content_list):
        item_type = item.get('type', '')

        # Track headers for PDF whiteout
        if item_type == 'header':
            metadata['headers_to_whiteout'].append({
                'index': i,
                'bbox': item.get('bbox'),
                'page_idx': item.get('page_idx'),
                'type': item_type,
            })

        # Skip before intro (title, authors, abstract)
        if i < intro_idx:
            metadata['removed_before_intro'].append({
                'index': i, 'bbox': item.get('bbox'),
                'page_idx': item.get('page_idx'), 'type': item_type,
            })
            continue

        # Skip after references
        if references_idx >= 0 and i >= references_idx:
            metadata['removed_after_refs'].append({
                'index': i, 'bbox': item.get('bbox'),
                'page_idx': item.get('page_idx'), 'type': item_type,
            })
            continue

        # Skip reproducibility
        if reproducibility_start >= 0 and reproducibility_start <= i < reproducibility_end:
            metadata['removed_reproducibility'].append({
                'index': i, 'bbox': item.get('bbox'),
                'page_idx': item.get('page_idx'), 'type': item_type,
            })
            continue

        # Skip acknowledgments
        if acknowledgments_start >= 0 and acknowledgments_start <= i < acknowledgments_end:
            metadata['removed_acknowledgments'].append({
                'index': i, 'bbox': item.get('bbox'),
                'page_idx': item.get('page_idx'), 'type': item_type,
            })
            continue

        # Skip first-page / URL-containing footnotes
        if item_type in ('page_footnote', 'footer'):
            page_idx = item.get('page_idx', 0)
            text = item.get('text', '')
            if page_idx == 0 or _has_url(text):
                metadata['removed_footnotes'].append({
                    'index': i, 'bbox': item.get('bbox'),
                    'page_idx': page_idx, 'type': item_type,
                })
                continue

        # Skip aside_text
        if item_type == 'aside_text':
            metadata['removed_aside_text'].append({
                'index': i, 'bbox': item.get('bbox'),
                'page_idx': item.get('page_idx'), 'type': item_type,
            })
            continue

        # Skip page 0 affiliation text
        page_idx = item.get('page_idx', 0)
        text = item.get('text', '')
        if page_idx == 0 and _is_affiliation_text(text):
            metadata['removed_affiliations'].append({
                'index': i, 'bbox': item.get('bbox'),
                'page_idx': page_idx, 'type': item_type,
                'text': text[:100] if text else '',
            })
            continue

        # Skip type="header" (tracked above for whiteout)
        if item_type == 'header':
            continue

        item_copy = copy.deepcopy(item)

        # Uppercase H1 headers
        if item_copy.get('text_level') == 1:
            item_copy['text'] = item_copy.get('text', '').upper()

        filtered.append(item_copy)

    # Note: paperprep doesn't use DEANONYMIZED mode (it always runs ANONYMOUS).
    # The abstract-anonymization helper that used to be wired in here lives in
    # the upstream simple-cli branch under paper_anonymizer.core.abstract.
    # If a caller of DEANONYMIZED mode actually needs anonymized-abstract
    # injection, pass an already-anonymized abstract_text — we just pass it
    # through verbatim as the markdown prefix.
    if abstract_text:
        metadata['markdown_prefix'] = f"\n{abstract_text}\n\n"

    return filtered, metadata
