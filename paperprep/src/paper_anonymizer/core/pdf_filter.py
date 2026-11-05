"""Unified PDF whiteout and trim logic for both anonymous and deanonymized modes.

Ported from scripts/generate_original_nips.py (NeurIPS implementation —
the cleanest reference). Both modes share the same page selection strategy:

- If refs exist and refs_page == last_body_page → keep the shared page,
  white-out refs content on it (mid-page case)
- If refs exist and refs_page != last_body_page → drop the refs page entirely
- If no refs → keep all pages

References are NEVER kept in the output (either trimmed as a page or whited out
if they share a page with body content).
"""

from paper_anonymizer.core.constants import (
    REFERENCES_PATTERN,
    ABSTRACT_PATTERN,
    ACKNOWLEDGMENTS_PATTERN,
    REPRODUCIBILITY_PATTERN,
)
from paper_anonymizer.core.filters import FilterMode


def _find_references_cutoff(content_list: list[dict]) -> tuple[int, int | None]:
    """Find the REFERENCES header index and page.

    Returns:
        (cutoff_idx, cutoff_page) or (len(content_list), None) if no refs.
    """
    for i, item in enumerate(content_list):
        if item.get('text_level') == 1:
            text = item.get('text', '').strip()
            if REFERENCES_PATTERN.match(text):
                return i, item.get('page_idx', 0)
    return len(content_list), None


def _compute_pages_to_keep(
    references_idx: int,
    references_page: int | None,
    last_page: int,
) -> int:
    """Determine how many pages to keep in the output PDF.

    Args:
        references_idx: Index of REFERENCES header (or len(content_list)).
        references_page: Page where refs start (or None).
        last_page: Page of last body item before refs.

    Returns:
        Number of pages to keep (pages 0..pages_to_keep-1).
    """
    if references_page is None:
        return 1000  # No refs — keep all (caller caps by len(doc))
    if references_page == last_page:
        return references_page + 1  # Shared mid-page: keep it
    return last_page + 1  # Refs on new page: don't include it


def compute_anonymous_whiteout(
    content_list: list[dict],
) -> tuple[list[dict], int]:
    """Compute whiteout items and pages_to_keep for ANONYMOUS mode.

    Matches collect_items_to_whiteout_minimal() in
    scripts/generate_original_nips.py.

    Whited out:
    - type="header" items on kept pages (running page titles)
    - type="aside_text" items on kept pages (line numbers, margin notes)
    - All content from cutoff_idx onwards IF refs start mid-page on the last body page

    Returns:
        (items_to_whiteout, pages_to_keep)
    """
    cutoff_idx, cutoff_page = _find_references_cutoff(content_list)

    # Last body page (page of last item before refs)
    if cutoff_idx > 0:
        last_page = content_list[cutoff_idx - 1].get('page_idx', 0)
    elif content_list:
        last_page = content_list[-1].get('page_idx', 0)
    else:
        last_page = 0

    pages_to_keep = _compute_pages_to_keep(cutoff_idx, cutoff_page, last_page)
    final_page = pages_to_keep - 1 if cutoff_page is not None else 1000

    items = []
    for i, item in enumerate(content_list):
        page_idx = item.get('page_idx', 0)
        if page_idx > final_page:
            continue

        item_type = item.get('type', '')
        bbox = item.get('bbox')
        if not bbox:
            continue

        # Whiteout headers and asides on kept pages
        if item_type == 'header':
            items.append(item)
            continue
        if item_type == 'aside_text':
            items.append(item)
            continue

        # Mid-page case: whiteout refs content on the shared page
        if cutoff_page is not None and cutoff_page == last_page and i >= cutoff_idx:
            if page_idx == cutoff_page:
                items.append(item)
                continue

    return items, pages_to_keep


def _find_abstract_intro_idx(content_list: list[dict]) -> tuple[int, int]:
    """Find abstract header and intro start indices.

    Returns:
        (abstract_idx, intro_idx). abstract_idx is -1 if not found.
        intro_idx is the first content item after the abstract.
    """
    abstract_idx = -1
    for i, item in enumerate(content_list):
        if item.get('text_level') == 1:
            text = item.get('text', '').strip()
            if ABSTRACT_PATTERN.match(text):
                abstract_idx = i
                break

    # intro is 2 items after abstract header (header + abstract text, then body)
    # matches generate_original_nips.py line 521
    intro_idx = abstract_idx + 2 if abstract_idx >= 0 else 0
    return abstract_idx, intro_idx


def _find_ack_repro_sections(
    content_list: list[dict],
    intro_idx: int,
    references_idx: int,
) -> tuple[int, int, int, int]:
    """Find acknowledgments and reproducibility section boundaries.

    Returns:
        (ack_start, ack_end, repro_start, repro_end). -1 if not found.
    """
    ack_start = ack_end = repro_start = repro_end = -1

    search_end = references_idx if references_idx < len(content_list) else len(content_list)

    for i in range(intro_idx, search_end):
        item = content_list[i]
        if item.get('text_level') != 1:
            continue
        text = item.get('text', '').strip()

        if ACKNOWLEDGMENTS_PATTERN.search(text):
            ack_start = i
            # Find end: next text_level=1 or end of search range
            for j in range(i + 1, search_end):
                if content_list[j].get('text_level') == 1:
                    ack_end = j
                    break
            else:
                ack_end = search_end
        elif REPRODUCIBILITY_PATTERN.search(text):
            repro_start = i
            for j in range(i + 1, search_end):
                if content_list[j].get('text_level') == 1:
                    repro_end = j
                    break
            else:
                repro_end = search_end

    return ack_start, ack_end, repro_start, repro_end


def compute_deanonymized_whiteout(
    content_list: list[dict],
) -> tuple[list[dict], int]:
    """Compute whiteout items and pages_to_keep for DEANONYMIZED mode.

    Matches collect_items_to_whiteout_clean() in
    scripts/generate_original_nips.py.

    Whited out:
    - All content before intro (title, authors, abstract)
    - type="header" + type="aside_text" on kept pages
    - Page-0 footnotes (page_footnote, footer)
    - Entire acknowledgments section
    - Entire reproducibility section
    - All content from references_idx onward on kept pages

    Returns:
        (items_to_whiteout, pages_to_keep)
    """
    cutoff_idx, cutoff_page = _find_references_cutoff(content_list)

    if cutoff_idx > 0:
        last_page = content_list[cutoff_idx - 1].get('page_idx', 0)
    elif content_list:
        last_page = content_list[-1].get('page_idx', 0)
    else:
        last_page = 0

    pages_to_keep = _compute_pages_to_keep(cutoff_idx, cutoff_page, last_page)
    final_page = pages_to_keep - 1 if cutoff_page is not None else 1000

    # Find intro and ack/repro boundaries
    _abstract_idx, intro_idx = _find_abstract_intro_idx(content_list)
    ack_start, ack_end, repro_start, repro_end = _find_ack_repro_sections(
        content_list, intro_idx, cutoff_idx
    )

    items = []
    for i, item in enumerate(content_list):
        page_idx = item.get('page_idx', 0)
        if page_idx > final_page:
            continue

        item_type = item.get('type', '')
        bbox = item.get('bbox')
        if not bbox:
            continue

        # Whiteout before intro (title, authors, abstract)
        if i < intro_idx:
            items.append(item)
            continue

        # Whiteout headers and asides
        if item_type in ('header', 'aside_text'):
            items.append(item)
            continue

        # Whiteout first-page footnotes
        if item_type in ('page_footnote', 'footer') and page_idx == 0:
            items.append(item)
            continue

        # Whiteout acknowledgments
        if ack_start >= 0 and ack_start <= i < ack_end:
            items.append(item)
            continue

        # Whiteout reproducibility
        if repro_start >= 0 and repro_start <= i < repro_end:
            items.append(item)
            continue

        # Whiteout refs content on kept pages (handles mid-page case)
        if cutoff_idx < len(content_list) and i >= cutoff_idx and page_idx <= final_page:
            items.append(item)
            continue

    return items, pages_to_keep


def compute_whiteout(
    content_list: list[dict],
    mode: FilterMode,
) -> tuple[list[dict], int]:
    """Unified entry point.

    Returns:
        (items_to_whiteout, pages_to_keep)
    """
    if mode == FilterMode.ANONYMOUS:
        return compute_anonymous_whiteout(content_list)
    else:
        return compute_deanonymized_whiteout(content_list)
