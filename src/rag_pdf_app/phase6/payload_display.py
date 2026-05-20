"""Coerce Qdrant payload fields for lab UI display.

Vector-store payloads are untrusted if the collection is shared, keys leak, or data is restored
from a backup. :func:`streamlit.markdown` interprets ``*``/``_``/``[`` etc.; never pass raw payload
strings into Markdown templates—only values that pass strict numeric coercion below.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from rag_pdf_app.phase6.models import PatchPlacement


def strict_uint_field(
    raw: object,
    *,
    fallback: int = 0,
    cap: int = 500_000,
) -> int:
    """Non-negative int suitable for grid indices; rejects bools, strings, and non-finite floats."""

    if raw is None or isinstance(raw, bool):
        return fallback
    n: int
    if isinstance(raw, int):
        n = raw
    elif isinstance(raw, float):
        if not math.isfinite(raw) or abs(raw - round(raw)) > 1e-9:
            return fallback
        n = int(round(raw))
    else:
        return fallback
    if n < 0 or n > cap:
        return fallback
    return n


def strict_float_metric(raw: object, *, fallback: float = 0.0) -> float:
    """Finite float for score display; clips to a modest range to avoid ``format`` surprises."""

    if raw is None:
        return fallback
    if isinstance(raw, bool):
        return fallback
    try:
        f = float(raw)
    except (TypeError, ValueError, OverflowError):
        return fallback
    if not math.isfinite(f):
        return fallback
    return float(max(-1e9, min(1e9, f)))


@dataclass(frozen=True, slots=True)
class SanitizedPhase6PatchDisplay:
    """Geometry pulled from Qdrant with rejects for non-integers and absurd magnitudes."""

    placement: PatchPlacement
    pixmap_page_width: int
    pixmap_page_height: int


def sanitized_phase6_patch_display(payload: Mapping[str, object]) -> SanitizedPhase6PatchDisplay:
    """Rebuild patch placement / pixmap dims for rerender + numeric caption lines."""

    p = PatchPlacement(
        page_index=strict_uint_field(payload.get("page_index"), fallback=0),
        row_index=strict_uint_field(payload.get("row_index"), fallback=0),
        col_index=strict_uint_field(payload.get("col_index"), fallback=0),
        x0_px=strict_uint_field(payload.get("x0_px"), fallback=0),
        y0_px=strict_uint_field(payload.get("y0_px"), fallback=0),
        width_px=max(1, strict_uint_field(payload.get("width_px"), fallback=1)),
        height_px=max(1, strict_uint_field(payload.get("height_px"), fallback=1)),
    )
    pw = max(1, strict_uint_field(payload.get("pixmap_page_width"), fallback=1))
    ph = max(1, strict_uint_field(payload.get("pixmap_page_height"), fallback=1))
    return SanitizedPhase6PatchDisplay(placement=p, pixmap_page_width=pw, pixmap_page_height=ph)
