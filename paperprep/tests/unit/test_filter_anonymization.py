"""Unit tests for ``stages.filter._verify_anonymized``.

The original heuristic looked only at headers + the first 10 blocks of page 0.
That missed title pages where the marker lands in a footer (lab style classes
that put the author block into ``\\fancyfoot[L]``) or after a banner of badges
exhausts the 10-block window.
"""
from __future__ import annotations

from paper_anonymizer.paperprep.stages.filter import _verify_anonymized


def _blk(page_idx, ty, text=""):
    return {"type": ty, "text": text, "page_idx": page_idx}


def test_anon_in_header_anywhere():
    cl = [_blk(0, "text", "VERO: a paper"), _blk(2, "header", "Under Review")]
    assert _verify_anonymized(cl) is True


def test_anon_in_footer_anywhere():
    """zlab.cls renders the (rewritten) corresponding-author block via
    ``\\fancyfoot[L]`` so the marker arrives as a ``type=footer`` block."""
    cl = [_blk(0, "text", "VERO: a paper"), _blk(0, "footer", "Anonymous Submission")]
    assert _verify_anonymized(cl) is True


def test_anon_late_on_page0_after_banner():
    """Banner-heavy title pages (project badges with images + short captions)
    used to push the author marker past the first-10-blocks cap. The marker
    must still be found anywhere on page 0."""
    cl = [_blk(0, "text", "TITLE")]
    cl += [_blk(0, "image", "")] * 8                    # banner images
    cl += [_blk(0, "text", "Models"), _blk(0, "text", "Data")]
    cl += [_blk(0, "text", "Code"), _blk(0, "text", "Project Page")]
    cl.append(_blk(0, "text", "Anonymous Submission"))  # block index 13
    cl.append(_blk(0, "text", "ABSTRACT"))
    assert _verify_anonymized(cl) is True


def test_no_marker_anywhere_fails():
    cl = [_blk(0, "text", "TITLE"), _blk(0, "text", "Joe Smith"),
          _blk(0, "text", "Princeton University"), _blk(0, "text", "ABSTRACT"),
          _blk(1, "footer", "1")]
    assert _verify_anonymized(cl) is False


def test_marker_only_on_later_page_fails():
    """A late-page marker (e.g. body text mentioning 'anonymous review') does
    NOT count -- the scan is restricted to page 0 + header/footer items."""
    cl = [_blk(0, "text", "TITLE"), _blk(2, "text", "we ran anonymous evaluations")]
    assert _verify_anonymized(cl) is False
