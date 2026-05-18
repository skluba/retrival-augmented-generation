"""Text cleaning and chunking (layout-aware via pdfminer spans, fallback to pypdf pages)."""

from __future__ import annotations

import hashlib
import io
import re
from typing import Any

from rag_pdf_app.parsing.models import TextSpan
from rag_pdf_app.parsing.relations import sorted_text_spans_for_layout
from rag_pdf_app.parsing.text_extractors import (
    extract_pdfminer_text_spans,
    extract_pypdf_metadata_and_plaintext,
)
from rag_pdf_app.rag.models import TextChunk


def clean_text(text: str) -> str:
    """Lightweight cleanup for extracted PDF strings."""

    t = text.replace("\x00", "")
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def _chunk_id_from(parts: list[str], page_start: int, page_end: int, source: str) -> str:
    blob = "\n".join(parts) + f"|{page_start}|{page_end}|{source}"
    return hashlib.sha256(blob.encode("utf-8", errors="replace")).hexdigest()[:24]


def _chunk_windows(text: str, chunk_size: int, overlap: int) -> list[str]:
    if not text:
        return []
    ov = min(overlap, max(chunk_size - 1, 0))
    step = max(chunk_size - ov, 1)
    out: list[str] = []
    i = 0
    while i < len(text):
        out.append(text[i : i + chunk_size])
        i += step
    return out


def _chunks_from_merged_layout_block(
    merged: str,
    *,
    buf_pages: list[int],
    source: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[TextChunk]:
    ps, pe = min(buf_pages), max(buf_pages)
    out: list[TextChunk] = []
    for segment in _chunk_windows(merged, chunk_size, chunk_overlap):
        seg = segment.strip()
        if not seg:
            continue
        cid = _chunk_id_from([seg], ps, pe, source)
        out.append(
            TextChunk(
                chunk_id=cid,
                text=seg,
                page_start=ps,
                page_end=pe,
                source=source,
                structure_note="pdfminer_lt_text_box_merge",
            )
        )
    return out


def chunks_from_layout_spans(
    spans: list[TextSpan],
    *,
    source: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[TextChunk]:
    """Merge pdfminer text boxes in reading order; split long runs with overlap."""

    ordered = sorted_text_spans_for_layout([s for s in spans if s.bbox is not None])
    chunks: list[TextChunk] = []
    buf_parts: list[str] = []
    buf_pages: list[int] = []

    def flush_buffer() -> None:
        if not buf_parts:
            return
        merged = clean_text("\n\n".join(buf_parts))
        if not merged:
            buf_parts.clear()
            buf_pages.clear()
            return
        chunks.extend(
            _chunks_from_merged_layout_block(
                merged,
                buf_pages=buf_pages,
                source=source,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )
        buf_parts.clear()
        buf_pages.clear()

    for sp in ordered:
        frag = clean_text(sp.text)
        if not frag:
            continue
        prospective = "\n\n".join([*buf_parts, frag]) if buf_parts else frag
        flush_first = len(prospective) > chunk_size and bool(buf_parts)
        if flush_first:
            flush_buffer()
        buf_parts.append(frag)
        buf_pages.append(sp.page_index)
    flush_buffer()
    return chunks


def chunks_from_pypdf_pages(
    per_page: dict[int, str],
    *,
    source: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[TextChunk]:
    """Flatten pages in order; chunk by sliding windows."""

    parts: list[str] = []
    page_boundaries: list[tuple[int, int]] = []
    pos = 0
    for idx in sorted(per_page.keys()):
        t = clean_text(per_page[idx])
        if not t:
            continue
        page_boundaries.append((pos, idx))
        parts.append(t)
        pos += len(t) + 2
    blob = "\n\n".join(parts)
    if not blob:
        return []

    def page_for_char_offset(off: int) -> int:
        current = page_boundaries[0][1]
        for start, pidx in page_boundaries:
            if start > off:
                break
            current = pidx
        return current

    chunks: list[TextChunk] = []
    ov = min(chunk_overlap, max(chunk_size - 1, 0))
    step = max(chunk_size - ov, 1)
    i = 0
    while i < len(blob):
        raw_seg = blob[i : i + chunk_size]
        seg = clean_text(raw_seg)
        if seg:
            mid = i + len(raw_seg) // 2
            pg = page_for_char_offset(mid)
            cid = _chunk_id_from([seg], pg, pg, source)
            chunks.append(
                TextChunk(
                    chunk_id=cid,
                    text=seg,
                    page_start=pg,
                    page_end=pg,
                    source=source,
                    structure_note="pypdf_page_flatten",
                )
            )
        i += step
    return chunks


def build_chunks_from_pdf_bytes(
    pdf_bytes: bytes,
    *,
    filename: str,
    chunk_size: int,
    chunk_overlap: int,
    use_layout: bool,
) -> tuple[list[TextChunk], list[str]]:
    """Return chunks plus ingestion notes (warnings / fallback reasons)."""

    notes: list[str] = []
    bio = io.BytesIO(pdf_bytes)
    if use_layout:
        spans = extract_pdfminer_text_spans(bio)
        layout_spans = [s for s in spans if s.bbox is not None]
        if layout_spans:
            ch = chunks_from_layout_spans(
                layout_spans,
                source=filename,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            if ch:
                return ch, notes
        notes.append("layout_chunking_empty_fallback_to_pypdf_pages")

    bio.seek(0)
    _, per_page = extract_pypdf_metadata_and_plaintext(bio)
    return (
        chunks_from_pypdf_pages(
            per_page,
            source=filename,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        ),
        notes,
    )


def chunks_to_langchain_payload(chunks: list[TextChunk]) -> tuple[list[str], list[dict[str, Any]]]:
    """Texts + metadata dicts for FAISS / Qdrant payloads."""

    texts = [c.text for c in chunks]
    metas: list[dict[str, Any]] = []
    for c in chunks:
        metas.append(
            {
                "chunk_id": c.chunk_id,
                "page_start": c.page_start,
                "page_end": c.page_end,
                "source": c.source,
                "structure_note": c.structure_note,
            }
        )
    return texts, metas
