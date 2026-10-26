"""Tests for PDF whiteout/trim logic — matches NeurIPS reference implementation."""

from paper_anonymizer.core.filters import FilterMode
from paper_anonymizer.core.pdf_filter import (
    compute_anonymous_whiteout,
    compute_deanonymized_whiteout,
    compute_whiteout,
    _find_references_cutoff,
    _compute_pages_to_keep,
)


def make_item(page_idx, item_type='text', text='', text_level=0, bbox=None):
    return {
        'type': item_type,
        'text': text,
        'text_level': text_level,
        'page_idx': page_idx,
        'bbox': bbox or [0, 0, 100, 100],
    }


class TestPageSelection:
    """Page selection logic — matches NeurIPS line 504-510."""

    def test_no_refs_keeps_all(self):
        assert _compute_pages_to_keep(10, None, 9) == 1000

    def test_refs_on_new_page_excludes_refs_page(self):
        # refs start on page 10, last body on page 9 -> keep pages 0-9
        assert _compute_pages_to_keep(10, 10, 9) == 10

    def test_refs_mid_page_keeps_shared_page(self):
        # refs start on page 9, last body on page 9 (mid-page) -> keep pages 0-9
        assert _compute_pages_to_keep(9, 9, 9) == 10


class TestFindReferencesCutoff:
    def test_finds_refs_header(self):
        cl = [
            make_item(0, text='Intro', text_level=1),
            make_item(1, text='body text'),
            make_item(5, text='REFERENCES', text_level=1),
            make_item(5, text='[1] foo et al'),
        ]
        cutoff_idx, cutoff_page = _find_references_cutoff(cl)
        assert cutoff_idx == 2
        assert cutoff_page == 5

    def test_no_refs(self):
        cl = [make_item(0, text='Intro', text_level=1)]
        cutoff_idx, cutoff_page = _find_references_cutoff(cl)
        assert cutoff_idx == 1
        assert cutoff_page is None

    def test_flexible_spacing(self):
        cl = [make_item(3, text='R E F E R E N C E S', text_level=1)]
        cutoff_idx, cutoff_page = _find_references_cutoff(cl)
        assert cutoff_idx == 0
        assert cutoff_page == 3


class TestAnonymousWhiteout:
    """Matches collect_items_to_whiteout_minimal() in generate_original_nips.py."""

    def test_headers_and_asides_whited_out(self):
        cl = [
            make_item(0, 'text', 'body'),
            make_item(0, 'header', 'Published at TMLR'),
            make_item(1, 'aside_text', '1'),  # line number
            make_item(1, 'text', 'more body'),
            make_item(2, 'text', 'REFERENCES', text_level=1),
            make_item(2, 'text', '[1] foo'),
        ]
        items, pages = compute_anonymous_whiteout(cl)
        # headers + asides whited out
        types = [item['type'] for item in items]
        assert 'header' in types
        assert 'aside_text' in types

    def test_refs_new_page_dropped(self):
        cl = [
            make_item(0, 'text', 'Intro', text_level=1),
            make_item(0, 'text', 'body'),
            make_item(1, 'text', 'more body'),
            make_item(2, 'text', 'REFERENCES', text_level=1),
            make_item(2, 'text', '[1] foo'),
        ]
        items, pages = compute_anonymous_whiteout(cl)
        # last body = page 1, refs start page 2 -> keep pages 0-1
        assert pages == 2
        # No refs items whited out (they're on a dropped page)
        ref_items = [i for i in items if i.get('page_idx') == 2]
        assert len(ref_items) == 0

    def test_refs_mid_page_whites_out_refs_on_shared_page(self):
        """Critical case: refs start mid-page, same page as last body."""
        cl = [
            make_item(0, 'text', 'Intro', text_level=1),
            make_item(0, 'text', 'body'),
            make_item(1, 'text', 'conclusion on page 1'),
            make_item(1, 'text', 'REFERENCES', text_level=1),  # same page as conclusion
            make_item(1, 'text', '[1] foo'),
            make_item(1, 'text', '[2] bar'),
        ]
        items, pages = compute_anonymous_whiteout(cl)
        # refs and last body both on page 1 -> keep pages 0-1 (2 pages)
        assert pages == 2
        # Refs header + refs content on page 1 should be whited out
        ref_text_items = [i for i in items if i.get('text', '').startswith('[') or 'REFERENCES' in i.get('text', '')]
        assert len(ref_text_items) == 3  # REFERENCES header + [1] foo + [2] bar


class TestDeanonymizedWhiteout:
    """Matches collect_items_to_whiteout_clean() in generate_original_nips.py."""

    def test_title_abstract_whited_out(self):
        cl = [
            make_item(0, 'text', 'Paper Title'),
            make_item(0, 'text', 'Author Names'),
            make_item(0, 'text', 'ABSTRACT', text_level=1),
            make_item(0, 'text', 'This paper...'),
            make_item(0, 'text', 'INTRODUCTION', text_level=1),
            make_item(0, 'text', 'body'),
            make_item(1, 'text', 'REFERENCES', text_level=1),
            make_item(1, 'text', '[1] foo'),
        ]
        items, pages = compute_deanonymized_whiteout(cl)
        # Title and author text should be whited out (before intro)
        titles = [i for i in items if 'Title' in i.get('text', '') or 'Author' in i.get('text', '')]
        assert len(titles) >= 2

    def test_acknowledgments_whited_out(self):
        cl = [
            make_item(0, 'text', 'ABSTRACT', text_level=1),
            make_item(0, 'text', 'abstract text'),
            make_item(0, 'text', 'INTRODUCTION', text_level=1),
            make_item(0, 'text', 'body'),
            make_item(1, 'text', 'ACKNOWLEDGMENTS', text_level=1),
            make_item(1, 'text', 'We thank...'),
            make_item(2, 'text', 'REFERENCES', text_level=1),
            make_item(2, 'text', '[1]'),
        ]
        items, pages = compute_deanonymized_whiteout(cl)
        ack_items = [i for i in items if 'ACKNOWLEDG' in i.get('text', '') or 'thank' in i.get('text', '')]
        assert len(ack_items) >= 2

    def test_refs_mid_page_whites_out_refs(self):
        cl = [
            make_item(0, 'text', 'ABSTRACT', text_level=1),
            make_item(0, 'text', 'abstract'),
            make_item(0, 'text', 'INTRODUCTION', text_level=1),
            make_item(0, 'text', 'body'),
            make_item(1, 'text', 'conclusion'),
            make_item(1, 'text', 'REFERENCES', text_level=1),  # mid-page refs
            make_item(1, 'text', '[1]'),
        ]
        items, pages = compute_deanonymized_whiteout(cl)
        assert pages == 2  # keep shared page 1
        ref_items = [i for i in items if 'REFERENCES' in i.get('text', '') or i.get('text', '') == '[1]']
        assert len(ref_items) == 2


class TestUnifiedEntryPoint:
    def test_anonymous_mode_dispatch(self):
        cl = [make_item(0, 'header', 'page header'), make_item(0, 'text', 'body')]
        items, _ = compute_whiteout(cl, FilterMode.ANONYMOUS)
        # header should be whited out
        assert any(i['type'] == 'header' for i in items)

    def test_deanonymized_mode_dispatch(self):
        cl = [
            make_item(0, 'text', 'title'),
            make_item(0, 'text', 'ABSTRACT', text_level=1),
            make_item(0, 'text', 'abstract'),
            make_item(0, 'text', 'INTRODUCTION', text_level=1),
            make_item(0, 'text', 'body'),
        ]
        items, _ = compute_whiteout(cl, FilterMode.DEANONYMIZED)
        # title should be whited out (before intro)
        assert any(i.get('text') == 'title' for i in items)
