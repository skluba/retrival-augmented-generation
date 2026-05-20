"""Tests for Phase 5.1 table chunk materialisation."""

from __future__ import annotations

from rag_pdf_app.parsing.models import BBox, ParsedPdf, TableBlock
from rag_pdf_app.rag.models import TextChunk
from rag_pdf_app.rag.table_chunks import (
    merge_narrative_and_table_chunks,
    table_text_chunks_from_parsed_pdf,
)


def _minimal_table() -> TableBlock:
    return TableBlock(
        table_id="t1",
        page_index=2,
        bbox=BBox(page_index=2, x0=0.0, y0=0.0, x1=100.0, y1=100.0),
        extractor="pymupdf_find_tables",
        rows=[["A", "B"], ["1", "2"]],
        as_markdown="| A | B |\n|---|---|\n| 1 | 2 |",
        as_csv="A,B\n1,2\n",
        summary="Tiny grid",
    )


def test_table_chunk_content_and_metadata() -> None:
    parsed = ParsedPdf(
        filename="x.pdf",
        pdf_bytes_sha256="a" * 64,
        tables=[_minimal_table()],
    )
    chunks = table_text_chunks_from_parsed_pdf(parsed)
    assert len(chunks) == 1
    c = chunks[0]
    assert c.chunk_kind == "pdf_table"
    assert c.content_type == "table_structured"
    assert c.table_id == "t1"
    assert c.table_csv_preview is not None and "A,B" in c.table_csv_preview
    assert "TABLE" in c.text and "Tiny grid" in c.text


def test_merge_appends_tables() -> None:
    n = TextChunk(
        chunk_id="n1",
        text="hello",
        page_start=0,
        page_end=0,
        source="x.pdf",
    )
    parsed = ParsedPdf(filename="x.pdf", pdf_bytes_sha256="b" * 64, tables=[_minimal_table()])
    t = table_text_chunks_from_parsed_pdf(parsed)[0]
    merged, mode = merge_narrative_and_table_chunks([n], [t])
    assert mode == "merged"
    assert len(merged) == 2
    assert merged[0].chunk_kind == "narrative"
    assert merged[1].chunk_kind == "pdf_table"


def test_merge_skips_when_no_tables() -> None:
    n = TextChunk(
        chunk_id="n1",
        text="hello",
        page_start=0,
        page_end=0,
        source="x.pdf",
    )
    merged, mode = merge_narrative_and_table_chunks([n], [])
    assert mode == "narrative_only"
    assert merged == [n]
