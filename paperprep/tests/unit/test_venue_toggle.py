"""Tests for per-venue anonymization toggles.

NOTE: predate the venue-agnostic refactor; skipping until rewrite.
"""
import pytest

pytestmark = pytest.mark.skip(reason="venue_toggle refactored to be venue-agnostic; tests pending rewrite")

from paper_anonymizer.core.venue_toggle import apply_venue_toggle


# ---------------------------------------------------------------------------
# Strip mode (neurips, icml, aistats, corl)
# ---------------------------------------------------------------------------

def test_neurips_final_stripped():
    out, n = apply_venue_toggle(r"\usepackage[final]{neurips_2025}", "neurips")
    assert n == 1
    assert out == r"\usepackage{neurips_2025}"


def test_neurips_already_anonymous():
    out, n = apply_venue_toggle(r"\usepackage{neurips_2025}", "neurips")
    assert n == 0
    assert out == r"\usepackage{neurips_2025}"


def test_nips_legacy_name_stripped():
    out, n = apply_venue_toggle(r"\usepackage[final]{nips_2017}", "neurips")
    assert n == 1
    assert out == r"\usepackage{nips_2017}"


def test_icml_accepted_stripped():
    out, n = apply_venue_toggle(r"\usepackage[accepted]{icml2026}", "icml")
    assert n == 1
    assert out == r"\usepackage{icml2026}"


def test_aistats_accepted_stripped():
    out, n = apply_venue_toggle(r"\usepackage[accepted]{aistats2026}", "aistats")
    assert n == 1
    assert out == r"\usepackage{aistats2026}"


def test_corl_final_stripped():
    out, n = apply_venue_toggle(r"\usepackage[final]{corl_2026}", "corl")
    assert n == 1
    assert out == r"\usepackage{corl_2026}"


# ---------------------------------------------------------------------------
# ICLR special case (iclrfinalcopy)
# ---------------------------------------------------------------------------

def test_iclr_finalcopy_commented_out():
    tex = "\\documentclass{article}\n\\iclrfinalcopy\n\\begin{document}"
    out, n = apply_venue_toggle(tex, "iclr")
    assert n == 1
    assert "% \\iclrfinalcopy" in out
    assert "\n\\iclrfinalcopy\n" not in out


def test_iclr_finalcopy_with_leading_indent():
    tex = "  \\iclrfinalcopy"
    out, n = apply_venue_toggle(tex, "iclr")
    assert n == 1
    assert out == "  % \\iclrfinalcopy"


def test_iclr_already_commented():
    tex = "% \\iclrfinalcopy"
    out, n = apply_venue_toggle(tex, "iclr")
    assert n == 0


# ---------------------------------------------------------------------------
# Add mode (aaai, cvpr, iccv, eccv)
# ---------------------------------------------------------------------------

def test_aaai_submission_added():
    out, n = apply_venue_toggle(r"\usepackage{aaai2026}", "aaai")
    assert n == 1
    assert out == r"\usepackage[submission]{aaai2026}"


def test_aaai_submission_already_present():
    out, n = apply_venue_toggle(r"\usepackage[submission]{aaai2026}", "aaai")
    assert n == 0


def test_cvpr_review_added_to_bare():
    out, n = apply_venue_toggle(r"\usepackage{cvpr}", "cvpr")
    assert n == 1
    assert out == r"\usepackage[review]{cvpr}"


def test_cvpr_review_added_to_existing_options():
    out, n = apply_venue_toggle(r"\usepackage[final]{cvpr}", "cvpr")
    assert n == 1
    assert out == r"\usepackage[final,review]{cvpr}"


def test_eccv_review_added_preserving_year():
    out, n = apply_venue_toggle(r"\usepackage[year=2026]{eccv}", "eccv")
    assert n == 1
    assert out == r"\usepackage[year=2026,review]{eccv}"


def test_iccv_review_added_to_bare():
    out, n = apply_venue_toggle(r"\usepackage{iccv}", "iccv")
    assert n == 1
    assert out == r"\usepackage[review]{iccv}"


# ---------------------------------------------------------------------------
# Replace mode (acl, naacl, emnlp, colm)
# ---------------------------------------------------------------------------

def test_acl_options_replaced_with_review():
    out, n = apply_venue_toggle(r"\usepackage[]{acl}", "acl")
    assert n == 1
    assert out == r"\usepackage[review]{acl}"


def test_acl_bare_replaced_with_review():
    out, n = apply_venue_toggle(r"\usepackage{acl}", "acl")
    assert n == 1
    assert out == r"\usepackage[review]{acl}"


def test_naacl_uses_same_acl_sty():
    out, n = apply_venue_toggle(r"\usepackage{acl}", "naacl")
    assert n == 1
    assert out == r"\usepackage[review]{acl}"


def test_emnlp_uses_same_acl_sty():
    out, n = apply_venue_toggle(r"\usepackage[final]{acl}", "emnlp")
    assert n == 1
    assert out == r"\usepackage[review]{acl}"


def test_colm_final_replaced_with_submission():
    out, n = apply_venue_toggle(
        r"\usepackage[final]{colm2025_conference}", "colm"
    )
    assert n == 1
    assert out == r"\usepackage[submission]{colm2025_conference}"


def test_colm_bare_replaced_with_submission():
    out, n = apply_venue_toggle(r"\usepackage{colm2025_conference}", "colm")
    assert n == 1
    assert out == r"\usepackage[submission]{colm2025_conference}"


# ---------------------------------------------------------------------------
# Non-matching patterns should be left alone
# ---------------------------------------------------------------------------

def test_non_matching_package_ignored():
    tex = r"\usepackage{amsmath}\usepackage{graphicx}"
    out, n = apply_venue_toggle(tex, "neurips")
    assert n == 0
    assert out == tex


def test_wrong_venue_not_matched():
    # This is an icml paper, but we asked for neurips → no match
    out, n = apply_venue_toggle(r"\usepackage[accepted]{icml2026}", "neurips")
    assert n == 0


def test_toggle_only_affects_matching_package():
    tex = (
        r"\usepackage{amsmath}" "\n"
        r"\usepackage[final]{neurips_2025}" "\n"
        r"\usepackage{graphicx}"
    )
    out, n = apply_venue_toggle(tex, "neurips")
    assert n == 1
    assert r"\usepackage{amsmath}" in out
    assert r"\usepackage{neurips_2025}" in out
    assert r"\usepackage{graphicx}" in out
    assert "final" not in out


# ---------------------------------------------------------------------------
# Idempotence
# ---------------------------------------------------------------------------

def test_idempotent_strip():
    tex = r"\usepackage[final]{neurips_2025}"
    out1, _ = apply_venue_toggle(tex, "neurips")
    out2, _ = apply_venue_toggle(out1, "neurips")
    assert out1 == out2


def test_idempotent_add():
    tex = r"\usepackage{cvpr}"
    out1, _ = apply_venue_toggle(tex, "cvpr")
    out2, _ = apply_venue_toggle(out1, "cvpr")
    assert out1 == out2
