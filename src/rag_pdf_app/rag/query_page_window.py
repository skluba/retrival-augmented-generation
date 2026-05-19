"""Strip inline page-window constraints from the query string (1-based inclusive pages)."""

from __future__ import annotations

import re

_PAGE_WINDOW_RE = re.compile(
    r"\b(?:only\s+)?(?:on\s+)?pages?\s*(\d+)\s*(?:-|–|to)\s*(\d+)\b",
    re.IGNORECASE,
)


def strip_inline_page_window(query: str) -> tuple[str, tuple[int, int] | None]:
    """Remove ``pages 10-20`` style clauses for retrieval; return (clean_query, (lo, hi) or None).

    Bounds are **1-based inclusive** human page numbers matching analyst language.
    """

    m = _PAGE_WINDOW_RE.search(query)
    if not m:
        return query, None
    lo, hi = int(m.group(1)), int(m.group(2))
    if lo > hi:
        lo, hi = hi, lo
    cleaned = _PAGE_WINDOW_RE.sub(" ", query)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned, (lo, hi)


def inline_window_to_zero_based(win: tuple[int, int] | None) -> tuple[int, int] | None:
    if win is None:
        return None
    lo1, hi1 = win
    return lo1 - 1, hi1 - 1
