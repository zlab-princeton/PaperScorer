"""LaTeX-source scrub — mirrors core.filters deanonymized-mode behavior.

Applies four rewrites, reusing the same regex patterns as the markdown
content-list filter so TeX and markdown scrubbing stay in sync:

    1. Remove `\\section{Acknowledg...}` / `\\section*{Acknowledg...}` blocks
    2. Remove `\\section{Repro...}` / `\\section*{Repro...}` blocks
    3. Delete `\\footnote{...}` whose body contains a URL or affiliation marker
    4. Strip `\\url{...}` and unwrap `\\href{url}{text}` → `text`
"""
from pathlib import Path
import re

from paper_anonymizer.core.constants import (
    ACKNOWLEDGMENTS_PATTERN,
    AFFILIATION_PATTERNS,
    CHECKLIST_PATTERN,
    REPRODUCIBILITY_PATTERN,
    URL_SENTENCE_PATTERN,
)

# Matches `\section{`, `\section*{`, `\subsection{`, `\subsection*{`,
# `\subsubsection*{`, `\paragraph{}`, with an optional `[short]` arg. Needed
# because some papers put Acknowledgments under `\subsection*{}` or similar.
_SECTION_OPEN = re.compile(
    r"\\(?:section|subsection|subsubsection|paragraph|subparagraph)\*?"
    r"\s*(?:\[[^\]]*\]\s*)?\{"
)
_END_DOCUMENT = re.compile(r"\\end\s*\{\s*document\s*\}")
# Structural commands that also terminate a \section{}'s scope — important
# because acknowledgments/reproducibility sections are often the LAST section
# before `\bibliography{}` or `\appendix`, and we must not eat those.
_SECTION_BOUNDARY = re.compile(
    r"\\(?:bibliography\b|bibliographystyle\b|printbibliography\b|"
    r"appendix\b|subfile\b|input\s*\{\s*(?:bib|refs?|bibliography|appendix)[^}]*\})"
)
_FOOTNOTE_OPEN = re.compile(r"\\footnote\s*\{")
_URL_OPEN = re.compile(r"\\url\s*\{")
_HREF_OPEN = re.compile(r"\\href\s*\{")
# AAAI's custom \link{text}{url} command — args REVERSED from \href.
_LINK_OPEN = re.compile(r"\\link\s*\{")
# Bare URLs in body text (no macro wrapper). Stop at whitespace or LaTeX
# structural chars. Trailing sentence punctuation like `.`/`,` is preserved
# via ``_strip_bare_url`` below.
_BARE_URL = re.compile(r"https?://[^\s\{\}\)\]\$]+")


def _strip_bare_url(m: re.Match) -> str:
    """Remove a bare URL but preserve trailing sentence punctuation."""
    url = m.group()
    trimmed = url.rstrip(".,;:!?")
    return url[len(trimmed):]  # the stripped trailing punctuation, if any

# Fragile author/affiliation commands whose contents are hidden in anonymous
# (review/submission) mode — the venue toggle forces this mode before compile,
# so URLs/footnotes inside these blocks are never rendered. We skip scrubbing
# inside them to avoid damaging the argument (e.g. leaving blank lines that
# trigger "Paragraph ended before \author was complete").
_ROBUST_CMD_OPENS = [
    re.compile(rf"\\{name}\s*\{{") for name in
    ("author", "institute", "thanks", "affil", "affiliation")
]


def _find_balanced_brace(text: str, open_idx: int) -> int:
    """Given ``open_idx`` pointing at a ``{``, return the index of the matching
    ``}``. Depth-aware; ignores ``\\{``, ``\\}`` and other ``\\<char>`` escapes.
    Returns -1 if unmatched (malformed source)."""
    n = len(text)
    if open_idx >= n or text[open_idx] != "{":
        return -1
    depth = 0
    i = open_idx
    while i < n:
        c = text[i]
        if c == "\\" and i + 1 < n:
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def remove_section(tex: str, header_pattern: re.Pattern) -> tuple[str, int]:
    """Remove every ``\\section{H}`` / ``\\section*{H}`` whose header ``H``
    matches ``header_pattern``, together with its content up to the next
    section or ``\\end{document}``. Returns ``(new_tex, n_removed)``.
    """
    out: list[str] = []
    i = 0
    removed = 0
    n = len(tex)
    while i < n:
        m = _SECTION_OPEN.search(tex, i)
        if not m:
            out.append(tex[i:])
            break
        brace_start = m.end() - 1
        brace_end = _find_balanced_brace(tex, brace_start)
        if brace_end == -1:
            out.append(tex[i:])
            break
        header = tex[brace_start + 1 : brace_end]
        if not header_pattern.search(header):
            out.append(tex[i : brace_end + 1])
            i = brace_end + 1
            continue

        # Section matches — keep prefix, skip to next section / bibliography /
        # appendix / end{document}. Must include bibliography commands or we'd
        # eat them when the removed section is the last before \end{document}.
        out.append(tex[i : m.start()])
        after = brace_end + 1
        nxt_sec = _SECTION_OPEN.search(tex, after)
        nxt_end = _END_DOCUMENT.search(tex, after)
        nxt_bnd = _SECTION_BOUNDARY.search(tex, after)
        cuts = [x.start() for x in (nxt_sec, nxt_end, nxt_bnd) if x]
        i = min(cuts) if cuts else n
        removed += 1

    return "".join(out), removed


_INLINE_ACK_LABEL = re.compile(
    r"(?:\\noindent\s*)?"
    r"\\(?:textbf|textit|textsc|emph|paragraph|subparagraph)\*?"
    r"\s*\{[Aa]cknowledg[^}]*\}[.:]?",
)
_PARA_BREAK = re.compile(r"\n\s*\n")


def remove_inline_acks(tex: str) -> tuple[str, int]:
    """Remove inline-style Acknowledgments paragraphs like
    ``\\textbf{Acknowledgment.} This work supported by...`` where the
    acknowledgment is NOT a `\\section{}` but an inline bold/italic label
    followed by paragraph text. Deletes from the label through the next
    paragraph break, section, bibliography, or end{document}."""
    out: list[str] = []
    i = 0
    n = len(tex)
    removed = 0
    while i < n:
        m = _INLINE_ACK_LABEL.search(tex, i)
        if not m:
            out.append(tex[i:])
            break
        # Skip commented-out matches.
        line_start = tex.rfind("\n", 0, m.start()) + 1
        if "%" in tex[line_start:m.start()]:
            out.append(tex[i:m.end()])
            i = m.end()
            continue
        out.append(tex[i:m.start()])
        # Find end: next paragraph break / section / bibliography / end{document}
        end = m.end()
        cuts = []
        for pat in (_PARA_BREAK, _SECTION_OPEN, _SECTION_BOUNDARY, _END_DOCUMENT):
            mm = pat.search(tex, end)
            if mm:
                cuts.append(mm.start())
        i = min(cuts) if cuts else n
        removed += 1
    return "".join(out), removed


def _body_is_leaky(body: str) -> bool:
    if URL_SENTENCE_PATTERN.search(body):
        return True
    return any(p.search(body) for p in AFFILIATION_PATTERNS)


def _protected_ranges(tex: str) -> list[tuple[int, int]]:
    """Return ``[(start, end_exclusive), ...]`` byte ranges covering the body
    of every ``\\author{}`` / ``\\institute{}`` / ``\\thanks{}`` / ``\\affil{}``
    / ``\\affiliation{}`` on the page. The range spans from the opening ``\\cmd``
    through the matching ``}`` (inclusive end index + 1). Ranges with unmatched
    braces are skipped. Returned list is sorted and non-overlapping."""
    ranges: list[tuple[int, int]] = []
    for pat in _ROBUST_CMD_OPENS:
        for m in pat.finditer(tex):
            brace_start = m.end() - 1
            brace_end = _find_balanced_brace(tex, brace_start)
            if brace_end == -1:
                continue
            ranges.append((m.start(), brace_end + 1))
    ranges.sort()
    return ranges


def _in_protected(pos: int, protected: list[tuple[int, int]] | None) -> int:
    """If ``pos`` falls inside a protected range, return that range's end index
    (to skip past). Otherwise return -1."""
    if not protected:
        return -1
    for start, end in protected:
        if start <= pos < end:
            return end
        if start > pos:
            break
    return -1


def remove_leaky_footnotes(
    tex: str, protected: list[tuple[int, int]] | None = None
) -> tuple[str, int]:
    """Delete every ``\\footnote{body}`` where ``body`` matches
    ``URL_SENTENCE_PATTERN`` or any ``AFFILIATION_PATTERNS``. Matches whose
    ``\\footnote`` falls inside a ``protected`` range are left alone.
    Returns ``(new_tex, n_removed)``.
    """
    out: list[str] = []
    i = 0
    removed = 0
    n = len(tex)
    while i < n:
        m = _FOOTNOTE_OPEN.search(tex, i)
        if not m:
            out.append(tex[i:])
            break
        prot_end = _in_protected(m.start(), protected)
        if prot_end > 0:
            out.append(tex[i:prot_end])
            i = prot_end
            continue
        brace_start = m.end() - 1
        brace_end = _find_balanced_brace(tex, brace_start)
        if brace_end == -1:
            out.append(tex[i:])
            break
        body = tex[brace_start + 1 : brace_end]
        if _body_is_leaky(body):
            out.append(tex[i : m.start()])
            removed += 1
        else:
            out.append(tex[i : brace_end + 1])
        i = brace_end + 1
    return "".join(out), removed


def strip_url_commands(
    tex: str, protected: list[tuple[int, int]] | None = None
) -> tuple[str, int, int]:
    """Apply two rewrites:

        - ``\\url{x}`` → (deleted)
        - ``\\href{url}{text}`` → ``text``

    Matches whose command-open falls inside a ``protected`` range are left
    alone. Returns ``(new_tex, urls_stripped, hrefs_unwrapped)``.
    """
    # Pass 1: \url{...}
    out: list[str] = []
    i = 0
    urls = 0
    n = len(tex)
    while i < n:
        m = _URL_OPEN.search(tex, i)
        if not m:
            out.append(tex[i:])
            break
        prot_end = _in_protected(m.start(), protected)
        if prot_end > 0:
            out.append(tex[i:prot_end])
            i = prot_end
            continue
        brace_start = m.end() - 1
        brace_end = _find_balanced_brace(tex, brace_start)
        if brace_end == -1:
            out.append(tex[i:])
            break
        out.append(tex[i : m.start()])
        i = brace_end + 1
        urls += 1
    tex = "".join(out)
    # Protected offsets are stale after pass 1 if any \url was removed.
    # But \href removal inside protected ranges is equally undesirable, so
    # recompute protected ranges for pass 2.
    protected = _protected_ranges(tex) if protected else None

    # Pass 2: \href{url}{text} → text
    out = []
    i = 0
    hrefs = 0
    n = len(tex)
    while i < n:
        m = _HREF_OPEN.search(tex, i)
        if not m:
            out.append(tex[i:])
            break
        prot_end = _in_protected(m.start(), protected)
        if prot_end > 0:
            out.append(tex[i:prot_end])
            i = prot_end
            continue
        brace1_start = m.end() - 1
        brace1_end = _find_balanced_brace(tex, brace1_start)
        if brace1_end == -1:
            out.append(tex[i:])
            break
        # Skip whitespace between the two argument groups.
        j = brace1_end + 1
        while j < n and tex[j] in " \t\r\n":
            j += 1
        if j >= n or tex[j] != "{":
            # No second arg — leave the command alone, advance past first arg.
            out.append(tex[i : brace1_end + 1])
            i = brace1_end + 1
            continue
        brace2_start = j
        brace2_end = _find_balanced_brace(tex, brace2_start)
        if brace2_end == -1:
            out.append(tex[i:])
            break
        out.append(tex[i : m.start()])
        out.append(tex[brace2_start + 1 : brace2_end])
        i = brace2_end + 1
        hrefs += 1
    tex = "".join(out)
    protected = _protected_ranges(tex) if protected else None

    # Pass 3: \link{text}{url} → text (AAAI custom, args REVERSED vs \href).
    out = []
    i = 0
    n = len(tex)
    while i < n:
        m = _LINK_OPEN.search(tex, i)
        if not m:
            out.append(tex[i:])
            break
        prot_end = _in_protected(m.start(), protected)
        if prot_end > 0:
            out.append(tex[i:prot_end])
            i = prot_end
            continue
        brace1_start = m.end() - 1
        brace1_end = _find_balanced_brace(tex, brace1_start)
        if brace1_end == -1:
            out.append(tex[i:])
            break
        j = brace1_end + 1
        while j < n and tex[j] in " \t\r\n":
            j += 1
        if j >= n or tex[j] != "{":
            out.append(tex[i : brace1_end + 1])
            i = brace1_end + 1
            continue
        brace2_start = j
        brace2_end = _find_balanced_brace(tex, brace2_start)
        if brace2_end == -1:
            out.append(tex[i:])
            break
        # Keep the first arg (text), drop the second (url).
        out.append(tex[i : m.start()])
        out.append(tex[brace1_start + 1 : brace1_end])
        i = brace2_end + 1
        hrefs += 1
    tex = "".join(out)

    # Pass 4: bare URLs in body text (https://... not inside \url/\href/\link).
    # Running this LAST means \url{} / \href{} / \link{} have already been
    # processed — any URL left is a bare body URL.
    tex, url_bare = _BARE_URL.subn(_strip_bare_url, tex)
    urls += url_bare

    return tex, urls, hrefs


def scrub_tex(tex: str) -> tuple[str, dict]:
    """Apply all four rewrites in order. Returns ``(new_tex, stats)``.

    URL/footnote rewrites skip content inside ``\\author{}`` / ``\\institute{}``
    / ``\\thanks{}`` / ``\\affil{}`` / ``\\affiliation{}`` blocks — venue toggle
    forces anonymous mode, so those arguments are hidden in the rendered PDF
    and don't need scrubbing. Skipping them avoids damaging the fragile
    arguments (which would otherwise trigger "Paragraph ended" errors)."""
    tex, ack_n = remove_section(tex, ACKNOWLEDGMENTS_PATTERN)
    tex, inline_ack_n = remove_inline_acks(tex)
    ack_n += inline_ack_n
    tex, repro_n = remove_section(tex, REPRODUCIBILITY_PATTERN)
    tex, checklist_n = remove_section(tex, CHECKLIST_PATTERN)
    protected = _protected_ranges(tex)
    tex, foot_n = remove_leaky_footnotes(tex, protected=protected)
    # Footnote removal shifts offsets, so recompute before URL/href pass.
    protected = _protected_ranges(tex)
    tex, url_n, href_n = strip_url_commands(tex, protected=protected)
    return tex, {
        "ack_sections_removed": ack_n,
        "repro_sections_removed": repro_n,
        "checklist_sections_removed": checklist_n,
        "leaky_footnotes_removed": foot_n,
        "urls_stripped": url_n,
        "hrefs_unwrapped": href_n,
    }


def scrub_tex_file(path: Path) -> dict:
    """Scrub a single .tex file in place. Returns stats (same keys as
    :func:`scrub_tex`)."""
    original = path.read_text(encoding="utf-8", errors="replace")
    scrubbed, stats = scrub_tex(original)
    if scrubbed != original:
        path.write_text(scrubbed, encoding="utf-8")
    return stats


def scrub_tex_dir(tex_dir: Path) -> dict:
    """Scrub every ``*.tex`` under ``tex_dir`` (recursive). Returns aggregated
    stats plus ``files_changed``. Bibliography files (``.bbl`` / ``.bib``) are
    left alone — URLs there are citations to cited works, not author-identifying."""
    agg = {
        "files_changed": 0,
        "ack_sections_removed": 0,
        "repro_sections_removed": 0,
        "checklist_sections_removed": 0,
        "leaky_footnotes_removed": 0,
        "urls_stripped": 0,
        "hrefs_unwrapped": 0,
    }
    for tex_file in sorted(tex_dir.rglob("*.tex")):
        stats = scrub_tex_file(tex_file)
        if any(stats.values()):
            agg["files_changed"] += 1
            for k, v in stats.items():
                agg[k] += v
    return agg
