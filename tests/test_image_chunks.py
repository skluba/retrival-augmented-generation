"""Tests for Phase 5.2 image-derived RAG chunks."""

from __future__ import annotations

from rag_pdf_app.parsing.models import BBox, ImageBlock, ParsedPdf
from rag_pdf_app.rag.image_chunks import image_text_chunks_from_parsed_pdf


def _img(xref: int = 7, *, caption: str | None = "Revenue increases in 2023.") -> ImageBlock:
    return ImageBlock(
        xref=xref,
        page_index=1,
        bbox=BBox(page_index=1, x0=50.0, y0=60.0, x1=550.0, y1=460.0),
        mime_type="image/png",
        width_px=512,
        height_px=400,
        caption=caption,
        caption_model="gemini-2.0-flash",
        contextual_snippet_above="Figure 2. Operating performance",
        contextual_snippet_below="Source: Management discussion.",
    )


def test_image_chunk_uses_caption_and_snippets() -> None:
    parsed = ParsedPdf(filename="r.pdf", pdf_bytes_sha256="c" * 64, images=[_img()])
    chunks = image_text_chunks_from_parsed_pdf(parsed, min_area_px=1000)
    assert len(chunks) == 1
    c = chunks[0]
    assert c.chunk_kind == "pdf_image"
    assert c.content_type == "figure_visual"
    assert c.image_xref == 7
    assert "Revenue increases" in c.text
    assert "Operating performance" in c.text


def test_image_chunk_skips_when_tiny_and_empty() -> None:
    tiny = ImageBlock(
        xref=1,
        page_index=0,
        bbox=BBox(page_index=0, x0=0.0, y0=0.0, x1=10.0, y1=10.0),
        width_px=10,
        height_px=10,
        caption=None,
    )
    parsed = ParsedPdf(filename="r.pdf", pdf_bytes_sha256="d" * 64, images=[tiny])
    chunks = image_text_chunks_from_parsed_pdf(parsed, min_area_px=8192)
    assert chunks == []


def test_image_chunk_keeps_tiny_if_caption_present() -> None:
    tiny_c = ImageBlock(
        xref=2,
        page_index=0,
        bbox=BBox(page_index=0, x0=0.0, y0=0.0, x1=10.0, y1=10.0),
        width_px=8,
        height_px=8,
        caption="Legend box",
    )
    parsed = ParsedPdf(filename="r.pdf", pdf_bytes_sha256="e" * 64, images=[tiny_c])
    chunks = image_text_chunks_from_parsed_pdf(parsed, min_area_px=8192)
    assert len(chunks) == 1
