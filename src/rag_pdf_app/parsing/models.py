"""Typed structures for parsed PDF content (JSON-friendly for UI / downstream RAG)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class BBox(BaseModel):
    """PDF coordinates in PyMuPDF space: origin top-left, y increases downward."""

    page_index: int = Field(ge=0, description="0-based page index")
    x0: float
    y0: float
    x1: float
    y1: float


class TextSpan(BaseModel):
    """One logical text region from a layout-aware extractor."""

    span_id: str
    page_index: int
    bbox: BBox | None = None
    text: str
    extractor: Literal["pypdf_page", "pdfminer", "docling_markdown"]
    structure_hint: str | None = Field(
        default=None,
        description="Optional label, e.g. pdfminer lineage or Docling heading level.",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class ImageBlock(BaseModel):
    xref: int
    page_index: int
    bbox: BBox
    mime_type: str | None = None
    width_px: int | None = None
    height_px: int | None = None
    image_bytes_b64: str | None = Field(
        default=None,
        description="PNG/JPEG/WebP bytes encoded as base64 for UI/export (optional).",
    )
    caption: str | None = None
    caption_model: str | None = None
    extractor: Literal["pymupdf"] = "pymupdf"
    metadata: dict[str, Any] = Field(default_factory=dict)
    related_text_above_span_id: str | None = None
    related_text_below_span_id: str | None = None
    contextual_snippet_above: str | None = Field(
        default=None,
        description="Nearby text fused for caption prompting / debugging.",
    )
    contextual_snippet_below: str | None = None


class TableBlock(BaseModel):
    table_id: str
    page_index: int
    bbox: BBox | None = None
    extractor: Literal["camelot_lattice", "camelot_stream", "pymupdf_find_tables", "llm"]
    rows: list[list[str | float | None]]
    as_markdown: str | None = None
    as_csv: str | None = None
    as_html: str | None = None
    as_json: list[dict[str, Any]] | None = None
    summary: str | None = Field(
        default=None,
        description="Optional short synopsis when cells are noisy or truncated.",
    )
    summary_model: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    related_text_above_span_id: str | None = None
    related_text_below_span_id: str | None = None


class ParsedPdf(BaseModel):
    """Full parse artefact."""

    filename: str
    pdf_bytes_sha256: str
    file_level_metadata: dict[str, Any] = Field(default_factory=dict)
    pypdf_per_page_plaintext: dict[int, str] = Field(
        default_factory=dict,
        description="0-based page index -> flattened text via pypdf.",
    )
    text_spans: list[TextSpan] = Field(
        default_factory=list,
        description="pdfminer + optional Docling markdown + pypdf page snapshots.",
    )
    docling_markdown: str | None = None
    images: list[ImageBlock] = Field(default_factory=list)
    tables: list[TableBlock] = Field(default_factory=list)
    parsing_notes: list[str] = Field(
        default_factory=list,
        description="Warnings such as extractor failures.",
    )
