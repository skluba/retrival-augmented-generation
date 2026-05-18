"""Unit tests for Phase 1 chunking helpers."""

from rag_pdf_app.parsing.models import BBox, TextSpan
from rag_pdf_app.rag.chunking import (
    chunks_from_layout_spans,
    chunks_from_pypdf_pages,
    clean_text,
)


def test_clean_text_collapses_whitespace() -> None:
    assert clean_text("a  \n\n  b") == "a b"


def test_chunks_from_pypdf_pages_respects_windows() -> None:
    per_page = {0: "alpha " * 200, 1: "beta " * 200}
    chunks = chunks_from_pypdf_pages(
        per_page,
        source="x.pdf",
        chunk_size=80,
        chunk_overlap=10,
    )
    assert len(chunks) >= 2
    assert all(c.source == "x.pdf" for c in chunks)


def test_layout_spans_merge_and_split() -> None:
    spans = [
        TextSpan(
            span_id="p0-0",
            page_index=0,
            bbox=BBox(page_index=0, x0=0.0, y0=10.0, x1=100.0, y1=30.0),
            text="First paragraph content.",
            extractor="pdfminer",
        ),
        TextSpan(
            span_id="p0-1",
            page_index=0,
            bbox=BBox(page_index=0, x0=0.0, y0=40.0, x1=100.0, y1=60.0),
            text="Second block follows.",
            extractor="pdfminer",
        ),
    ]
    out = chunks_from_layout_spans(
        spans,
        source="y.pdf",
        chunk_size=40,
        chunk_overlap=5,
    )
    assert out
    assert out[0].structure_note == "pdfminer_lt_text_box_merge"
