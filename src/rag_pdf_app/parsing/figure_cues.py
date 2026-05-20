"""Heuristics to tell report-style numbered figures apart from stray embedded rasters."""

from __future__ import annotations

_FIG_NEEDLES = frozenset(
    (
        "figure ",
        "figure:",
        "figure.",
        "figures ",
        "figures:",
        "figures.",
        "fig.",
        "fig ",
    )
)
_COMPACT_PREFIXES = ("figure", "figures", "fig")


def _preceded_by_word_char(blob_lower: str, idx: int) -> bool:
    return idx > 0 and blob_lower[idx - 1].isalnum()


def _has_digit_after_punctuation(tail: str) -> bool:
    i = 0
    while i < len(tail) and tail[i] in " \t\r\n:.-":
        i += 1
    return bool(i < len(tail) and tail[i].isdigit())


def _scan_needles(blob: str) -> bool:
    """Match 'Figure N' style tokens separated by punctuation or spaces."""

    for needle in _FIG_NEEDLES:
        nl = len(needle)
        start = 0
        while True:
            i = blob.find(needle, start)
            if i == -1:
                break
            tail = blob[i + nl : i + nl + 24]
            if (
                not _preceded_by_word_char(blob, i)
                and _has_digit_after_punctuation(tail)
            ):
                return True
            start = i + nl
    return False


def _scan_compact_glue(blob: str) -> bool:
    """Match OCR-ish glue ('Figure12', 'Fig3')."""

    for compact in _COMPACT_PREFIXES:
        start = 0
        cl = len(compact)
        while True:
            i = blob.find(compact, start)
            if i == -1:
                break
            nxt = blob[i + cl : i + cl + 4]
            start = i + cl
            if (
                not _preceded_by_word_char(blob, i)
                and bool(nxt)
                and nxt[0].isdigit()
            ):
                return True
    return False


def nearby_text_has_explicit_figure_label(*snippets: str | None) -> bool:
    """True if concatenated snippets mention a numbered figure caption line."""

    blob = "\n".join((s or "").strip() for s in snippets if (s or "").strip())
    if not blob:
        return False
    low = blob[:8000].lower()
    return _scan_needles(low) or _scan_compact_glue(low)
