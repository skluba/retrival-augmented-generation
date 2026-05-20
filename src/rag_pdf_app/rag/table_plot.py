"""Lightweight plotting helpers for retrieved table CSV previews (Phase 5.1, Streamlit)."""

from __future__ import annotations

import io
import logging

import pandas as pd

from rag_pdf_app.config import Settings
from rag_pdf_app.rag.models import RetrievalHit

_LOG = logging.getLogger(__name__)


def dataframe_from_hits(
    hits: list[RetrievalHit],
    settings: Settings,
) -> tuple[pd.DataFrame | None, str | None]:
    """Parse the first hit that carries a bounded CSV preview in metadata.

    Returns ``(DataFrame, note)``. ``note`` explains skips for the UI.
    """

    if not settings.rag_plotting_enabled:
        return None, "plotting_disabled"

    for h in hits:
        raw = h.metadata.get("table_csv_preview")
        if not isinstance(raw, str) or not raw.strip():
            continue
        try:
            df = pd.read_csv(io.StringIO(raw.strip()))
        except Exception as exc:  # noqa: BLE001 — best-effort
            _LOG.debug("Plot helper CSV parse failed: %s", exc)
            continue
        if df.shape[0] == 0 or df.shape[1] == 0:
            continue
        return df, None

    return None, "no_tabular_preview"


def chartable_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Prefer numeric columns for ``st.bar_chart`` / ``st.line_chart``."""

    num = df.select_dtypes(include=["number"]).copy()
    if num.shape[1] > 0:
        return num
    out = pd.DataFrame(index=df.index)
    for col in df.columns:
        coerced = pd.to_numeric(df[col], errors="coerce")
        if coerced.notna().sum() >= max(2, int(0.5 * len(df))):
            out[col] = coerced
    return out.select_dtypes(include=["number"])
