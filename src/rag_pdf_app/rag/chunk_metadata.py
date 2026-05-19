"""Lightweight metadata cues derived from chunk text (Phase 3 retrieval hints)."""

from __future__ import annotations

import re


_FIGURE_RE = re.compile(
    r"\b(figure|fig\.|chart|graph|diagram)\b",
    re.IGNORECASE,
)
_TABLE_RE = re.compile(
    r"\b(table\s+\d+|schedule\s+[ivx\d]|million\b|billion\b|\$\s*[\d,]+)\b",
    re.IGNORECASE,
)


def infer_chunk_content_type(text: str, structure_note: str | None) -> str:
    """Coarse bucket for boosting / analytics (not a formal schema)."""

    sample = text[:800]
    low = sample.lower()
    if _FIGURE_RE.search(sample):
        return "figure_ref"
    if _TABLE_RE.search(sample) or (structure_note and "table" in structure_note.lower()):
        return "table_dense"
    if re.search(r"\$\s*[\d,]+|[\d,]+\s*million|[\d,]+\s*billion", sample):
        return "numeric_heavy"
    return "narrative"


def infer_section_hint(text: str, *, max_chars: int = 96) -> str | None:
    """First non-empty line as an approximate section / heading cue."""

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if len(line) > max_chars:
            return f"{line[: max_chars - 1]}…"
        return line
    return None
