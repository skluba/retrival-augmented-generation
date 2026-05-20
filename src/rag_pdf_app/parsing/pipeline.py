"""End-to-end PDF parsing: metadata, structured text, images, tables, Gemini enrichments."""

from __future__ import annotations

import base64
import contextlib
import hashlib
import io
import os
import tempfile

from rag_pdf_app.config import Settings
from rag_pdf_app.parsing.figure_cues import nearby_text_has_explicit_figure_label
from rag_pdf_app.parsing.images import extract_images_pymupdf
from rag_pdf_app.parsing.models import ImageBlock, ParsedPdf, TableBlock, TextSpan
from rag_pdf_app.parsing.relations import (
    nearest_neighbors,
    snippets_for_neighbor_ids,
    sorted_text_spans_for_layout,
)
from rag_pdf_app.parsing.tables import extract_tables_camelot, extract_tables_pymupdf
from rag_pdf_app.parsing.text_extractors import (
    extract_docling_markdown,
    extract_pdfminer_text_spans,
    extract_pypdf_metadata_and_plaintext,
)
from rag_pdf_app.vertex_gemini import caption_document_image, summarize_table_for_rag


def _spans_from_pypdf_pages(per_page: dict[int, str]) -> list[TextSpan]:
    text_spans: list[TextSpan] = []
    for page_idx, text in sorted(per_page.items()):
        stripped = text.strip()
        if not stripped:
            continue
        text_spans.append(
            TextSpan(
                span_id=f"pypdf-page-{page_idx}",
                page_index=page_idx,
                bbox=None,
                text=stripped,
                extractor="pypdf_page",
                metadata={"engine": "pypdf.Page.extract_text"},
            )
        )
    return text_spans


def _docling_markdown_span(doc_md: str) -> TextSpan:
    return TextSpan(
        span_id="docling-markdown-document",
        page_index=0,
        bbox=None,
        text=doc_md.strip(),
        extractor="docling_markdown",
        metadata={"source": "docling.export_to_markdown"},
    )


def _camelot_tables_via_tempfile(data: bytes, notes: list[str]) -> list[TableBlock]:
    tmp_pdf: str | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tf:
            tf.write(data)
            tf.flush()
            tmp_pdf = tf.name
        return extract_tables_camelot(tmp_pdf, notes)
    except Exception as exc:  # noqa: BLE001 — temp file paths / ghostscript quirks
        notes.append(f"camelot_path_error:{exc}")
        return []
    finally:
        if tmp_pdf:
            with contextlib.suppress(OSError):
                os.unlink(tmp_pdf)


def _neighbor_ids_fallback_page(
    tbl: TableBlock,
    layout_spines: list[TextSpan],
) -> tuple[str | None, str | None]:
    same_page = [
        sp for sp in layout_spines if sp.page_index == tbl.page_index and sp.bbox is not None
    ]
    aid = same_page[0].span_id if same_page else None
    bid = same_page[-1].span_id if len(same_page) > 1 else None
    return aid, bid


def _table_neighbor_ids(
    tbl: TableBlock,
    layout_spines: list[TextSpan],
) -> tuple[str | None, str | None]:
    if tbl.bbox is None:
        return _neighbor_ids_fallback_page(tbl, layout_spines)
    page_idx_tbl = tbl.bbox.page_index
    spans_same_page = [
        sp for sp in layout_spines if sp.bbox is not None and sp.bbox.page_index == page_idx_tbl
    ]
    return nearest_neighbors(tbl.bbox, spans_on_page=spans_same_page)


def _apply_table_neighbors(
    tbl: TableBlock,
    layout_spines: list[TextSpan],
    spans_by_id: dict[str, TextSpan],
) -> None:
    aid, bid = _table_neighbor_ids(tbl, layout_spines)
    snippets = snippets_for_neighbor_ids(aid, bid, spans_by_id=spans_by_id)
    tbl.related_text_above_span_id = aid
    tbl.related_text_below_span_id = bid
    tbl.metadata["contextual_snippet_above"] = snippets[0]
    tbl.metadata["contextual_snippet_below"] = snippets[1]


def _summarize_table_if_enabled(
    tbl: TableBlock,
    settings: Settings,
    *,
    gemini_table_summaries: bool,
    notes: list[str],
) -> None:
    if not gemini_table_summaries or not (tbl.as_markdown or tbl.as_csv):
        return
    try:
        summary = summarize_table_for_rag(settings, tbl.as_markdown, tbl.as_csv)
        if summary.strip():
            tbl.summary = summary.strip()
            tbl.summary_model = settings.vertex_generative_model
    except Exception as exc:  # noqa: BLE001 — Vertex quotas / OCR noise
        notes.append(f"gemini_table_summary_{tbl.table_id}:{exc}")


def _enrich_tables(
    tables_final: list[TableBlock],
    layout_spines: list[TextSpan],
    spans_by_id: dict[str, TextSpan],
    settings: Settings,
    *,
    gemini_table_summaries: bool,
    notes: list[str],
) -> None:
    for tbl in tables_final:
        _apply_table_neighbors(tbl, layout_spines, spans_by_id)
        _summarize_table_if_enabled(
            tbl,
            settings,
            gemini_table_summaries=gemini_table_summaries,
            notes=notes,
        )


def _apply_image_neighbors(
    img: ImageBlock,
    layout_spines: list[TextSpan],
    spans_by_id: dict[str, TextSpan],
) -> None:
    page_idx_img = img.bbox.page_index
    spans_same_page = [
        sp for sp in layout_spines if sp.bbox is not None and sp.bbox.page_index == page_idx_img
    ]
    aid, bid = nearest_neighbors(img.bbox, spans_on_page=spans_same_page)
    snippets = snippets_for_neighbor_ids(aid, bid, spans_by_id=spans_by_id)
    img.related_text_above_span_id = aid
    img.related_text_below_span_id = bid
    img.contextual_snippet_above = snippets[0]
    img.contextual_snippet_below = snippets[1]


def _caption_image_if_enabled(
    img: ImageBlock,
    settings: Settings,
    *,
    gemini_image_captions: bool,
    notes: list[str],
) -> None:
    if not gemini_image_captions or not img.image_bytes_b64:
        return
    try:
        blob = base64.b64decode(img.image_bytes_b64)
    except ValueError as exc:
        notes.append(f"img_decode_xref_{img.xref}:{exc}")
        return
    mime = img.mime_type or "image/png"
    try:
        img.caption = caption_document_image(
            settings,
            image_bytes=blob,
            mime_type=mime,
            neighbour_above=img.contextual_snippet_above,
            neighbour_below=img.contextual_snippet_below,
        )
        img.caption_model = settings.vertex_generative_model
    except Exception as exc:  # noqa: BLE001 — Vertex quotas / multimodal failures
        notes.append(f"gemini_img_caption_xref_{img.xref}:{exc}")


def _enrich_images(
    images_blocks: list[ImageBlock],
    layout_spines: list[TextSpan],
    spans_by_id: dict[str, TextSpan],
    settings: Settings,
    *,
    gemini_image_captions: bool,
    notes: list[str],
) -> None:
    kept: list[ImageBlock] = []
    for img in images_blocks:
        _apply_image_neighbors(img, layout_spines, spans_by_id)
        if settings.rag_image_require_figure_label_nearby and not nearby_text_has_explicit_figure_label(
            img.contextual_snippet_above,
            img.contextual_snippet_below,
        ):
            notes.append(
                f"img_dropped_no_figure_cue:xref={img.xref}:page={img.page_index + 1}"
            )
            continue
        _caption_image_if_enabled(
            img,
            settings,
            gemini_image_captions=gemini_image_captions,
            notes=notes,
        )
        kept.append(img)

    images_blocks.clear()
    images_blocks.extend(kept)


def parse_pdf_bytes(
    data: bytes,
    filename: str,
    *,
    settings: Settings,
    embed_image_base64: bool = True,
    gemini_image_captions: bool = True,
    gemini_table_summaries: bool = True,
    run_docling: bool = True,
    run_camelot: bool = False,
) -> ParsedPdf:
    """Parse ``data`` into a structured :class:`ParsedPdf`.

    ``run_camelot`` defaults to false because Camelot invokes Ghostscript on the PDF path,
    which is hazardous for untrusted uploads (native attack surface). PyMuPDF table detection
    still runs when Camelot is off.
    """
    notes: list[str] = []
    sha_hex = hashlib.sha256(data).hexdigest()

    bio = io.BytesIO(data)
    file_meta, per_page = extract_pypdf_metadata_and_plaintext(bio)

    bio.seek(0)
    miner_spans = extract_pdfminer_text_spans(bio)

    doc_md: str | None = None
    if run_docling:
        bio.seek(0)
        doc_md, doc_err = extract_docling_markdown(bio)
        if doc_err:
            notes.append(doc_err)

    text_spans = _spans_from_pypdf_pages(per_page)
    text_spans.extend(miner_spans)

    if doc_md and doc_md.strip():
        text_spans.append(_docling_markdown_span(doc_md))

    layout_spines = sorted_text_spans_for_layout([sp for sp in text_spans if sp.bbox is not None])
    spans_by_id = {sp.span_id: sp for sp in layout_spines}

    camelot_blocks: list[TableBlock] = []
    if run_camelot:
        camelot_blocks = _camelot_tables_via_tempfile(data, notes)
    else:
        notes.append(
            "camelot_skipped: Camelot/Ghostscript table extraction disabled "
            "(enable run_camelot only for trusted PDFs)."
        )
    pym_tables = extract_tables_pymupdf(data, notes)
    tables_final = [*camelot_blocks, *pym_tables]

    _enrich_tables(
        tables_final,
        layout_spines,
        spans_by_id,
        settings,
        gemini_table_summaries=gemini_table_summaries,
        notes=notes,
    )

    images_blocks = extract_images_pymupdf(data, embed_base64=embed_image_base64)
    notes.append(f"image_raster_candidates_raw:{len(images_blocks)}")
    _enrich_images(
        images_blocks,
        layout_spines,
        spans_by_id,
        settings,
        gemini_image_captions=gemini_image_captions,
        notes=notes,
    )

    return ParsedPdf(
        filename=filename,
        pdf_bytes_sha256=sha_hex,
        file_level_metadata=file_meta,
        pypdf_per_page_plaintext=per_page,
        text_spans=text_spans,
        docling_markdown=doc_md,
        images=images_blocks,
        tables=tables_final,
        parsing_notes=notes,
    )
