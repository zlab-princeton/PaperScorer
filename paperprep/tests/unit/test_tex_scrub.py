"""Tests for LaTeX-source scrubbing (`paper_anonymizer.core.tex`)."""
from paper_anonymizer.core.constants import (
    ACKNOWLEDGMENTS_PATTERN,
    REPRODUCIBILITY_PATTERN,
)
from paper_anonymizer.core.tex import (
    _find_balanced_brace,
    remove_leaky_footnotes,
    remove_section,
    scrub_tex,
    scrub_tex_dir,
    scrub_tex_file,
    strip_url_commands,
)


# ---------------------------------------------------------------------------
# Balanced-brace helper
# ---------------------------------------------------------------------------

def test_balanced_brace_simple():
    assert _find_balanced_brace("{abc}", 0) == 4


def test_balanced_brace_nested():
    assert _find_balanced_brace("{a{b}c}", 0) == 6


def test_balanced_brace_escaped():
    assert _find_balanced_brace(r"{a\{b\}c}", 0) == 8


def test_balanced_brace_unclosed():
    assert _find_balanced_brace("{abc", 0) == -1


# ---------------------------------------------------------------------------
# Section removal
# ---------------------------------------------------------------------------

def test_ack_section_removed():
    tex = (
        "Intro text. \\section{Acknowledgments}\n"
        "We thank funder X for support.\n"
        "\\section{References}\n"
        "\\bibliography{refs}\n"
    )
    out, n = remove_section(tex, ACKNOWLEDGMENTS_PATTERN)
    assert n == 1
    assert "Acknowledgments" not in out
    assert "thank funder" not in out
    assert "\\section{References}" in out


def test_ack_star_section_removed():
    tex = (
        "Body. \\section*{Acknowledgements}\n"
        "We thank X.\n"
        "\\section*{References}\n"
    )
    out, n = remove_section(tex, ACKNOWLEDGMENTS_PATTERN)
    assert n == 1
    assert "Acknowledgements" not in out
    assert "\\section*{References}" in out


def test_repro_section_removed():
    tex = (
        "\\section{Reproducibility Statement}\n"
        "All code is at our repo.\n"
        "\\section{Conclusion}\n"
    )
    out, n = remove_section(tex, REPRODUCIBILITY_PATTERN)
    assert n == 1
    assert "Reproducibility" not in out
    assert "All code" not in out
    assert "\\section{Conclusion}" in out


def test_section_without_match_kept():
    tex = "\\section{Introduction}\nThis is body."
    out, n = remove_section(tex, ACKNOWLEDGMENTS_PATTERN)
    assert n == 0
    assert out == tex


def test_section_extends_to_end_of_document():
    tex = (
        "\\section{Intro} body. \\section{Acknowledgments}\n"
        "Thanks everyone.\n"
        "\\end{document}\n"
    )
    out, n = remove_section(tex, ACKNOWLEDGMENTS_PATTERN)
    assert n == 1
    assert "Thanks everyone" not in out
    assert "\\end{document}" in out
    assert "\\section{Intro}" in out


def test_section_with_optional_short_title():
    tex = (
        "\\section[Ack]{Acknowledgments}\nThanks.\n\\section{References}\n"
    )
    out, n = remove_section(tex, ACKNOWLEDGMENTS_PATTERN)
    assert n == 1
    assert "Thanks" not in out
    assert "\\section{References}" in out


# ---------------------------------------------------------------------------
# Leaky footnote removal
# ---------------------------------------------------------------------------

def test_footnote_with_github_url_removed():
    tex = "See \\footnote{Code at https://github.com/foo/bar} for code."
    out, n = remove_leaky_footnotes(tex)
    assert n == 1
    assert "\\footnote" not in out
    assert "github" not in out


def test_footnote_with_nested_url_command_removed():
    tex = "Body\\footnote{See \\url{https://github.com/foo}}."
    out, n = remove_leaky_footnotes(tex)
    assert n == 1
    assert "\\footnote" not in out
    assert "github" not in out


def test_footnote_with_email_removed():
    tex = "Header\\footnote{Correspondence: alice@foo.edu}."
    out, n = remove_leaky_footnotes(tex)
    assert n == 1
    assert "@foo.edu" not in out


def test_footnote_with_equal_contribution_removed():
    tex = "Alice\\footnote{$^*$Equal contribution}, Bob."
    out, n = remove_leaky_footnotes(tex)
    assert n == 1
    assert "Equal contribution" not in out


def test_footnote_with_correspondence_removed():
    tex = "Title\\footnote{Correspondence to alice.}."
    out, n = remove_leaky_footnotes(tex)
    assert n == 1
    assert "Correspondence" not in out


def test_footnote_without_leak_kept():
    tex = "Body\\footnote{See Section 3 for details.}."
    out, n = remove_leaky_footnotes(tex)
    assert n == 0
    assert out == tex


def test_footnote_with_nested_braces():
    tex = "Body\\footnote{literal \\{brace\\} and https://github.com/foo}."
    out, n = remove_leaky_footnotes(tex)
    assert n == 1
    assert "\\footnote" not in out


def test_multiple_footnotes_mixed():
    tex = (
        "A\\footnote{https://x/y} B\\footnote{kept} "
        "C\\footnote{alice@x.edu} D"
    )
    out, n = remove_leaky_footnotes(tex)
    assert n == 2
    assert "kept" in out
    assert "https" not in out
    assert "@x.edu" not in out


# ---------------------------------------------------------------------------
# URL command stripping
# ---------------------------------------------------------------------------

def test_url_command_stripped():
    tex = "See \\url{https://github.com/foo/bar} for details."
    out, urls, hrefs = strip_url_commands(tex)
    assert urls == 1
    assert hrefs == 0
    assert "\\url" not in out
    assert "github" not in out


def test_href_unwrapped():
    tex = "See \\href{https://github.com/foo}{our codebase} here."
    out, urls, hrefs = strip_url_commands(tex)
    assert urls == 0
    assert hrefs == 1
    assert "\\href" not in out
    assert "github" not in out
    assert "our codebase" in out


def test_href_with_whitespace():
    tex = "See \\href { https://x } { our code } done."
    out, urls, hrefs = strip_url_commands(tex)
    assert hrefs == 1
    assert "\\href" not in out
    assert "our code" in out


def test_href_with_nested_braces_in_text():
    tex = "\\href{https://x}{our \\texttt{codebase}}"
    out, urls, hrefs = strip_url_commands(tex)
    assert hrefs == 1
    assert out == "our \\texttt{codebase}"


def test_multiple_urls():
    tex = "A \\url{https://x/y} B \\url{https://a/b} C"
    out, urls, hrefs = strip_url_commands(tex)
    assert urls == 2
    assert hrefs == 0
    assert "\\url" not in out


# ---------------------------------------------------------------------------
# End-to-end scrub_tex
# ---------------------------------------------------------------------------

def test_full_scrub_combined_fixture():
    tex = (
        "\\section{Intro} Intro body. "
        "See \\footnote{See https://github.com/x}"
        " and \\url{https://a/b} and \\url{https://c/d} and \\url{https://e/f}"
        " and \\href{https://g}{link}.\n"
        "\\section{Acknowledgments}\n"
        "Thanks.\n"
        "\\section{Reproducibility}\n"
        "Repro note.\n"
        "\\section{References}\n"
    )
    out, stats = scrub_tex(tex)
    assert stats == {
        "ack_sections_removed": 1,
        "repro_sections_removed": 1,
        "checklist_sections_removed": 0,
        "leaky_footnotes_removed": 1,
        "urls_stripped": 3,
        "hrefs_unwrapped": 1,
    }
    assert "Acknowledgments" not in out
    assert "Reproducibility" not in out
    assert "\\footnote" not in out
    assert "\\url" not in out
    assert "\\href" not in out
    assert "link" in out
    assert "\\section{References}" in out


def test_idempotent():
    tex = (
        "\\section{Acknowledgments} thanks. \\section{References}"
        " \\url{https://x} \\href{https://y}{z} \\footnote{https://q}"
    )
    out1, _ = scrub_tex(tex)
    out2, _ = scrub_tex(out1)
    assert out1 == out2


# ---------------------------------------------------------------------------
# Filesystem wrappers
# ---------------------------------------------------------------------------

def test_scrub_tex_file(tmp_path):
    p = tmp_path / "main.tex"
    p.write_text(
        "Body \\url{https://x/y}. \\section{Acknowledgments} thanks. "
        "\\section{References}"
    )
    stats = scrub_tex_file(p)
    assert stats["urls_stripped"] == 1
    assert stats["ack_sections_removed"] == 1
    new_content = p.read_text()
    assert "\\url" not in new_content
    assert "Acknowledgments" not in new_content
    assert "\\section{References}" in new_content


def test_scrub_tex_dir(tmp_path):
    (tmp_path / "a.tex").write_text("Body \\url{https://x}.")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.tex").write_text("\\href{https://y}{text}.")
    (tmp_path / "c.tex").write_text("No change here.")

    stats = scrub_tex_dir(tmp_path)
    assert stats["files_changed"] == 2
    assert stats["urls_stripped"] == 1
    assert stats["hrefs_unwrapped"] == 1
    # Untouched file stays untouched.
    assert (tmp_path / "c.tex").read_text() == "No change here."
