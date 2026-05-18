"""Typed chunks carried through ingestion and retrieval."""

from __future__ import annotations

from pydantic import BaseModel, Field


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
