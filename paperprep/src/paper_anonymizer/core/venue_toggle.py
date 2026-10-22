"""Universal LaTeX anonymization.

Every conference has its own `.sty` quirk for anonymizing — some use a
``[review]`` option, some a ``\\*finalcopy`` macro, some nothing at all.
Handling each one was brittle.

Instead, this module applies a single universal strategy that works for every
venue/year combination, inspired by ECCV's own official template (which
literally ships with ``\\author{Anonymous ECCV submission}`` hard-coded):

1. **Author/affiliation override (the source of truth).** Every un-commented
   ``\\author{...}`` / ``\\institute{...}`` / ``\\affiliations{...}`` /
   ``\\thanks{...}`` is rewritten to an anonymous stub. Valid LaTeX for every
   style, anonymizes every paper regardless of venue.

2. **Cosmetic cleanups (best-effort, idempotent).** We also attempt the
   style-native toggle — strip ``[final]``/``[accepted]`` options, comment
   out ``\\*finalcopy`` macros, add ``[review]``/``[submission]`` options to
   styles that accept them. These produce prettier PDFs with line numbers,
   "Under review" headers, etc. — but aren't required for anonymization.

The `venue` parameter is kept for backwards compatibility but no longer
branches behavior. Running the full pipeline on every paper catches all
cases with no per-venue dispatch.
"""
from __future__ import annotations

import re


_USEPACKAGE_RE    = re.compile(r"\\usepackage(\[[^\]]*\])?\s*\{([^}]+)\}")
_DOCUMENTCLASS_RE = re.compile(r"\\documentclass(\[[^\]]*\])?\s*\{([^}]+)\}")

# Open-brace detectors for author-like and affiliation-like commands.
# Covers generic LaTeX + venue-specific variants:
#   AISTATS: \aistatsauthor{}, \aistatsaddress{}
#   ICML:    \icmlauthor{}, \icmlauthorlist{}, \icmlaffiliation{}
#   ECCV:    \institute{}, \authorrunning{}, \instituterunning{}
#   general: \correspondingauthor{}, \authornote{}, \affiliations{}
# Match optional ``[superscript]`` arg between command and ``{``. Examples:
#   \author[1,2]{Name}   (authblk package style)
#   \affil[1]{Lab}
_OPT_ARG = r"\s*(?:\[[^\]]*\])?\s*"

_AUTHOR_OPEN = re.compile(
    r"\\(?:author|aistatsauthor|icmlauthor|icmlauthorlist|"
    r"authorrunning|correspondingauthor|authornote|coltauthor)" + _OPT_ARG + r"\{"
)
_AFFIL_OPEN = re.compile(
    r"\\(?:institute|instituterunning|affil(?:iations?)?|"
    r"aistatsaddress)" + _OPT_ARG + r"\{"
)
# Two-arg: \icmlaffiliation{LABEL}{TEXT} — replace only TEXT (arg 2) so the
# LABEL→TEXT lookup used by \icmlauthor{name}{labels} stays intact.
_ICML_AFFIL_OPEN = re.compile(r"\\icmlaffiliation" + _OPT_ARG + r"\{")
_THANKS_OPEN = re.compile(r"\\thanks" + _OPT_ARG + r"\{")

# Line-level finalcopy macros across conferences.
_FINALCOPY_RE = re.compile(
    r"^(?P<indent>[ \t]*)(?P<macro>\\(?:iclr|colm|acl|iccv|cvpr|eccv)finalcopy)\b",
    re.MULTILINE,
)

# Style name patterns used for option-level toggles.
_STRIP_STYLES = re.compile(
    r"^(?:(?:neurips|nips)_\d{4}|icml\d{4}|aistats\d{4}|corl_?\d{4})$"
)
_ADD_REVIEW_STYLES = re.compile(
    r"^(?:cvpr(?:\d{4})?|iccv(?:\d{4})?|eccv(?:\d{4})?|"
    r"(?:acl|emnlp|naacl)(?:\d{0,4})|EMNLP\d{4}|ACL\d{4})$"
)
_ADD_SUBMISSION_STYLES = re.compile(
    r"^(?:aaai(?:2[3-9]|[3-9]\d|202[3-9]|20[3-9]\d)|"
    r"colm\d{4}_conference)$"
)
_BAD_OPTIONS = {"final", "accepted", "preprint", "camera", "cameraready"}


# ──────────────────────────────────────────────────────────────────────────
# Brace-balanced helpers
# ──────────────────────────────────────────────────────────────────────────

def _find_balanced_brace(text: str, open_idx: int) -> int:
    """Given ``open_idx`` at a ``{``, return index of the matching ``}``.
    Depth-aware; ignores ``\\{``/``\\}`` escapes and LaTeX ``%`` comments.
    Returns -1 if unmatched."""
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
        if c == "%":
            nl = text.find("\n", i)
            i = n if nl == -1 else nl + 1
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _override_second_arg(tex: str, open_re: re.Pattern, body: str) -> tuple[str, int]:
    """For two-arg commands like ``\\icmlaffiliation{LABEL}{TEXT}``, replace
    only the SECOND ``{...}`` body. Keeps LABEL so cross-references resolve."""
    out: list[str] = []
    i = 0
    n = len(tex)
    count = 0
    while i < n:
        m = open_re.search(tex, i)
        if not m:
            out.append(tex[i:])
            break
        if _is_line_commented(tex, m.start()):
            nl = tex.find("\n", m.start())
            nl = n if nl == -1 else nl + 1
            out.append(tex[i:nl])
            i = nl
            continue
        b1_start = m.end() - 1
        b1_end = _find_balanced_brace(tex, b1_start)
        if b1_end == -1:
            out.append(tex[i : b1_start + 1])
            i = b1_start + 1
            continue
        # Skip whitespace between the two arg groups.
        j = b1_end + 1
        while j < n and tex[j] in " \t\r\n":
            j += 1
        if j >= n or tex[j] != "{":
            out.append(tex[i : b1_end + 1])
            i = b1_end + 1
            continue
        b2_start = j
        b2_end = _find_balanced_brace(tex, b2_start)
        if b2_end == -1:
            out.append(tex[i : b2_start + 1])
            i = b2_start + 1
            continue
        # Keep \cmd{LABEL}{, replace body, add closing }
        out.append(tex[i : b2_start + 1])
        out.append(body)
        out.append("}")
        i = b2_end + 1
        count += 1
    return "".join(out), count


def _is_line_commented(text: str, pos: int) -> bool:
    """True if an un-escaped ``%`` precedes ``pos`` on the same line."""
    line_start = text.rfind("\n", 0, pos) + 1
    i = line_start
    while i < pos:
        if text[i] == "\\" and i + 1 < pos:
            i += 2
            continue
        if text[i] == "%":
            return True
        i += 1
    return False


def _override_braced_cmd(tex: str, open_re: re.Pattern, body: str) -> tuple[str, int]:
    """Rewrite every un-commented ``\\cmd{...}`` (brace-balanced) by replacing
    the ``{...}`` body with ``body``. The command name is preserved so
    ``\\aistatsauthor{real}`` → ``\\aistatsauthor{Anonymous}``. Skips
    commented-out matches and malformed braces."""
    out: list[str] = []
    i = 0
    n = len(tex)
    count = 0
    while i < n:
        m = open_re.search(tex, i)
        if not m:
            out.append(tex[i:])
            break
        if _is_line_commented(tex, m.start()):
            nl = tex.find("\n", m.start())
            nl = n if nl == -1 else nl + 1
            out.append(tex[i:nl])
            i = nl
            continue
        brace_start = m.end() - 1
        brace_end = _find_balanced_brace(tex, brace_start)
        if brace_end == -1:
            out.append(tex[i : brace_start + 1])
            i = brace_start + 1
            continue
        # Keep everything up to the open brace (i.e. "\cmd{"), replace body.
        out.append(tex[i : brace_start + 1])
        out.append(body)
        out.append("}")
        i = brace_end + 1
        count += 1
    return "".join(out), count


# ──────────────────────────────────────────────────────────────────────────
# Option-level toggles
# ──────────────────────────────────────────────────────────────────────────

def _add_option(existing: str, new_option: str) -> str:
    """Add ``new_option`` to a comma-separated option list. Idempotent."""
    parts = [p.strip() for p in existing.split(",") if p.strip()]
    if new_option in parts:
        return ",".join(parts)
    parts.append(new_option)
    return ",".join(parts)


def _rewrite_options(tex: str, cmd: str) -> int:
    """Walk every ``\\cmd[opts]{style}`` and apply option transformations based
    on the style name. Returns total number of rewrites. `tex` is mutated via
    the returned tuple in the caller."""
    pat = _USEPACKAGE_RE if cmd == "usepackage" else _DOCUMENTCLASS_RE
    changes = [0]

    def repl(m: re.Match) -> str:
        opts_full = m.group(1) or ""
        style = m.group(2).strip()
        opts_inner = opts_full[1:-1] if opts_full else ""
        parts = [p.strip() for p in opts_inner.split(",") if p.strip()]
        new_parts = parts[:]

        # Universally safe: strip camera-ready options only. Adding [review]
        # or [submission] is SKIPPED because not every style declares those
        # options (e.g. older iccv.sty errors on unknown options). The
        # \author{} override above already anonymizes; this step only
        # removes explicit un-anonymization flags.
        if _STRIP_STYLES.match(style) or _ADD_REVIEW_STYLES.match(style) \
                or _ADD_SUBMISSION_STYLES.match(style):
            new_parts = [p for p in new_parts if p not in _BAD_OPTIONS]
        else:
            return m.group(0)

        if new_parts == parts:
            return m.group(0)

        changes[0] += 1
        if new_parts:
            return f"\\{cmd}[{','.join(new_parts)}]{{{style}}}"
        return f"\\{cmd}{{{style}}}"

    new_tex = pat.sub(repl, tex)
    return new_tex, changes[0]


# ──────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────

def apply_venue_toggle(tex: str, venue: str = "") -> tuple[str, int]:
    """Apply universal anonymization. Returns ``(new_tex, n_changes)``.

    The ``venue`` argument is accepted for backwards compatibility but no
    longer influences behavior — the same pipeline runs on every paper:

    1. Rewrite ``\\author{}`` / ``\\institute{}`` / ``\\affiliations{}`` /
       ``\\thanks{}`` to anonymous stubs (universal hard anonymization).
    2. Comment out ``\\*finalcopy`` macros (line-level).
    3. Strip ``[final]``/``[accepted]`` from NeurIPS-family usepackages.
    4. Add ``[review]`` to CVPR/ICCV/ECCV/ACL-family usepackages and
       documentclasses.
    5. Add ``[submission]`` to AAAI 2023+ and COLM usepackages.

    Each step is a no-op if its patterns don't match.
    """
    total = 0

    # Step 1: hard anonymization. Command name is preserved; only ``{...}``
    # body is replaced, so venue-specific commands like \aistatsauthor{} and
    # \icmlauthor{} still compile with their style.
    tex, n = _override_braced_cmd(tex, _AUTHOR_OPEN, "Anonymous Submission")
    total += n
    tex, n = _override_braced_cmd(tex, _AFFIL_OPEN, "Anonymous Institute")
    total += n
    # \icmlaffiliation{LABEL}{TEXT} — replace TEXT only, keep LABEL so
    # \icmlauthor{name}{LABEL} references still resolve.
    tex, n = _override_second_arg(tex, _ICML_AFFIL_OPEN, "Anonymous Institute")
    total += n
    # \thanks usually nested inside \author (already replaced); rare standalones.
    tex, n = _override_braced_cmd(tex, _THANKS_OPEN, "")
    total += n

    # Step 2: comment out finalcopy macros (ICLR / COLM / ACL / ICCV / CVPR / ECCV).
    tex, n = _FINALCOPY_RE.subn(r"\g<indent>% \g<macro>", tex)
    total += n

    # Step 3 & 4 & 5: option-level toggles for both \usepackage and \documentclass.
    tex, n = _rewrite_options(tex, "usepackage")
    total += n
    tex, n = _rewrite_options(tex, "documentclass")
    total += n

    return tex, total
