"""Spatial proximity between layout blocks using reading order."""

from __future__ import annotations

from rag_pdf_app.parsing.models import BBox, TextSpan


def sorted_text_spans_for_layout(spans: list[TextSpan]) -> list[TextSpan]:
    """Sort spans by page (asc) then vertical position (top to bottom)."""
    keyed: list[tuple[tuple[int, float, float, str], TextSpan]] = []
    for span in spans:
        if span.bbox is None:
            keyed.append(((span.page_index, float("inf"), float("inf"), span.span_id), span))
        else:
            b = span.bbox
            keyed.append(((b.page_index, b.y0, b.x0, span.span_id), span))
    keyed.sort(key=lambda pair: pair[0])
    return [pair[1] for pair in keyed]


def nearest_neighbors(
    anchor: BBox,
    *,
    spans_on_page: list[TextSpan],
) -> tuple[str | None, str | None]:
    """Return `(above_span_id, below_span_id)` on the same ``page_index``.

    Reading order follows PyMuPDF coords: smaller ``y`` sits higher on the page.
    """

    tol = 2.0
    page = anchor.page_index
    at_top = min(anchor.y0, anchor.y1)
    at_bot = max(anchor.y0, anchor.y1)

    above_id: str | None = None
    below_id: str | None = None
    best_above_gap = float("inf")
    best_below_gap = float("inf")

    for sp in spans_on_page:
        b = sp.bbox
        if b is None or b.page_index != page:
            continue
        tb_top = min(b.y0, b.y1)
        tb_bot = max(b.y0, b.y1)

        if tb_bot <= at_top + tol:
            gap = at_top - tb_bot
            if gap < best_above_gap:
                best_above_gap = gap
                above_id = sp.span_id

        if tb_top >= at_bot - tol:
            gap = tb_top - at_bot
            if gap < best_below_gap:
                best_below_gap = gap
                below_id = sp.span_id

    return above_id, below_id


def snippets_for_neighbor_ids(
    above_id: str | None,
    below_id: str | None,
    *,
    spans_by_id: dict[str, TextSpan],
    max_chars: int = 400,
) -> tuple[str | None, str | None]:
    """Pull compact neighbour text snippets for Gemini prompts."""

    def clip(t: str) -> str:
        t = t.strip()
        if len(t) <= max_chars:
            return t
        return f"{t[: max_chars - 15]}...[truncated]"

    above_t = spans_by_id[above_id].text if above_id and above_id in spans_by_id else None
    below_t = spans_by_id[below_id].text if below_id and below_id in spans_by_id else None

    return (clip(above_t) if above_t else None, clip(below_t) if below_t else None)
