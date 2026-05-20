"""Turn parsed PDF raster images into embeddable text chunks (Phase 5.2).

Chunks are built from Gemini figure captions (when enabled) plus neighbouring PDF text, so
semantic and BM25 retrieval can answer chart / graph questions without shipping raw pixels into
the vector store body.
"""

from __future__ import annotations

import hashlib

from rag_pdf_app.parsing.models import BBox, ImageBlock, ParsedPdf
from rag_pdf_app.rag.models import TextChunk


def _bbox_pixel_area(bbox: BBox) -> int:
    w = max(float(bbox.x1) - float(bbox.x0), 0.0)
    h = max(float(bbox.y1) - float(bbox.y0), 0.0)
    return int(w * h)


def _image_pixel_area(img: ImageBlock) -> int:
    if img.width_px and img.height_px:
        return max(int(img.width_px) * int(img.height_px), 0)
    return _bbox_pixel_area(img.bbox)


def _chunk_id_for_image(img: ImageBlock, pdf_digest: str) -> str:
    blob = f"{pdf_digest}\x00img\x00{img.xref}\x00{img.page_index}".encode()
    return hashlib.sha256(blob).hexdigest()[:24]


def _should_index_image(
    *,
    area: int,
    caption: str,
    above: str,
    below: str,
    min_area_px: int,
) -> bool:
    if not (caption or above or below):
        return False
    return not (area < min_area_px and not caption)


def _compose_image_chunk_body(
    img: ImageBlock,
    *,
    area: int,
    caption: str,
    above: str,
    below: str,
) -> str:
    w = img.width_px or "?"
    h = img.height_px or "?"
    parts: list[str] = [
        f"FIGURE · page {img.page_index + 1} · xref {img.xref} · {img.mime_type or 'image'}",
        f"Approximate size: {w}×{h}px (area ≈ {area})",
    ]
    if caption:
        model = img.caption_model or "unknown"
        parts.append(f"Visual description (model: {model}):\n{caption}")
    if above:
        parts.append(f"Nearby document text (above figure): {above[:1200]}")
    if below:
        parts.append(f"Nearby document text (below figure): {below[:1200]}")
    return "\n\n".join(parts).strip()


def image_text_chunk_from_block(
    img: ImageBlock,
    *,
    pdf_sha256: str,
    source: str,
    min_area_px: int,
    max_text_chars: int = 12_000,
) -> TextChunk | None:
    """Build one searchable chunk for an :class:`ImageBlock`, or ``None`` if skipped."""

    area = _image_pixel_area(img)
    caption = (img.caption or "").strip()
    above = (img.contextual_snippet_above or "").strip()
    below = (img.contextual_snippet_below or "").strip()
    # Avoid indexing boilerplate-only blobs: large figures without captions/snippets dominated
    # dense/BM25 pools with repeated generic text and diluted eval retrieval (Phase 5.2).
    if not _should_index_image(
        area=area,
        caption=caption,
        above=above,
        below=below,
        min_area_px=min_area_px,
    ):
        return None

    body = _compose_image_chunk_body(img, area=area, caption=caption, above=above, below=below)
    if not body:
        return None

    cid = _chunk_id_for_image(img, pdf_sha256)
    section = caption.split("\n", 1)[0][:240] if caption else f"figure page {img.page_index + 1}"

    return TextChunk(
        chunk_id=cid,
        text=body[:max_text_chars],
        page_start=img.page_index,
        page_end=img.page_index,
        source=source,
        structure_note="pdf_image:pymupdf",
        content_type="figure_visual",
        section_hint=section,
        chunk_kind="pdf_image",
        image_xref=img.xref,
    )


def image_text_chunks_from_parsed_pdf(
    parsed: ParsedPdf,
    *,
    min_area_px: int,
    max_text_chars: int = 12_000,
) -> list[TextChunk]:
    """Materialise at most one chunk per extracted image that passes the filter."""

    out: list[TextChunk] = []
    for img in parsed.images:
        ch = image_text_chunk_from_block(
            img,
            pdf_sha256=parsed.pdf_bytes_sha256,
            source=parsed.filename,
            min_area_px=min_area_px,
            max_text_chars=max_text_chars,
        )
        if ch is not None:
            out.append(ch)
    return out
