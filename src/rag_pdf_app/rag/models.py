"""Typed chunks carried through ingestion and retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field


@dataclass
class RetrievalHit:
    """One retrieved segment with scores and LangChain-style metadata."""

    chunk_id: str
    text: str
    score: float
    metadata: dict[str, Any]


class TextChunk(BaseModel):
    """One searchable segment derived from PDF text."""

    chunk_id: str
    text: str
    page_start: int = Field(ge=0, description="0-based first page index covered")
    page_end: int = Field(ge=0, description="0-based last page index covered")
    source: str = Field(description="Original filename or URI label")
    structure_note: str | None = Field(
        default=None,
        description="How this chunk was formed (e.g. merged pdfminer boxes).",
    )
    content_type: str = Field(
        default="narrative",
        description="Coarse label for filtering/boosting (narrative, table_dense, figure_ref, …).",
    )
    section_hint: str | None = Field(
        default=None,
        description="Approximate heading / first-line cue for display and light retrieval boosts.",
    )
