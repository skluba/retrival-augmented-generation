"""Text + metadata extraction using pypdf, pdfminer, and optional Docling."""

from __future__ import annotations

import contextlib
import io
import os
import tempfile
from typing import Any, BinaryIO

from pdfminer.high_level import extract_pages
from pdfminer.layout import LAParams, LTTextBox
from pypdf import PdfReader

from rag_pdf_app.parsing.models import BBox, TextSpan


def _safe_primitive(v: Any) -> Any:
    if v is None:
        return None
    try:
        if hasattr(v, "get_object"):
            return _safe_primitive(v.get_object())
    except Exception:  # noqa: BLE001
        return str(v)
    if isinstance(v, (str, int, float, bool)):
        return v
    if isinstance(v, bytes):
        return v.decode(errors="replace")
    try:
        from datetime import datetime

        if isinstance(v, datetime):
            return v.isoformat()
    except Exception:
        pass
    return str(v)


def extract_pypdf_metadata_and_plaintext(stream: BinaryIO) -> tuple[dict[str, Any], dict[int, str]]:
    raw = stream.read()
    stream.seek(0)

    fp = io.BytesIO(raw)

    reader = PdfReader(fp)
    meta_src = reader.metadata or {}
    meta: dict[str, Any] = {}
    for k, v in meta_src.items():
        key = str(k).lstrip("/")
        meta[key] = _safe_primitive(v)

    per_page: dict[int, str] = {}
    for idx, page in enumerate(reader.pages):
        try:
            per_page[idx] = page.extract_text() or ""
        except Exception:
            per_page[idx] = ""
    return meta, per_page


def _iter_lt_text_boxes(component: Any) -> Any:
    if isinstance(component, LTTextBox):
        yield component
    for child in getattr(component, "_objs", []) or []:
        yield from _iter_lt_text_boxes(child)


def extract_pdfminer_text_spans(stream: BinaryIO) -> list[TextSpan]:
    laparams = LAParams(
        line_margin=0.15,
        char_margin=1.75,
        word_margin=0.2,
        boxes_flow=0.5,
    )

    raw = stream.read()
    stream.seek(0)
    pdf_fp = io.BytesIO(raw)

    spans: list[TextSpan] = []
    for page_index, layout in enumerate(extract_pages(pdf_fp, laparams=laparams)):
        for shape_idx, item in enumerate(_iter_lt_text_boxes(layout)):
            txt = item.get_text().strip()
            if not txt:
                continue
            (x0, y0, x1, y1) = item.bbox
            spans.append(
                TextSpan(
                    span_id=f"pdfminer-p{page_index}-{shape_idx}",
                    page_index=page_index,
                    bbox=BBox(
                        page_index=page_index,
                        x0=float(x0),
                        y0=float(y0),
                        x1=float(x1),
                        y1=float(y1),
                    ),
                    text=txt,
                    extractor="pdfminer",
                    structure_hint=type(item).__name__,
                    metadata={"pdfminer_type": type(item).__name__},
                )
            )

    return spans


def extract_docling_markdown(stream: BinaryIO) -> tuple[str | None, str | None]:
    """Return `(markdown, error)` where error is populated on failure/disabled."""

    try:
        from docling.document_converter import DocumentConverter
    except Exception as exc:  # noqa: BLE001
        return None, f"docling_import_failed: {exc}"

    data = stream.read()
    stream.seek(0)

    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(data)
            tmp.flush()
            tmp_path = tmp.name

        converter = DocumentConverter()
        result = converter.convert(tmp_path)

        doc = getattr(result, "document", None)
        if doc is None:
            return None, "docling_no_document"
        exporter = getattr(doc, "export_to_markdown", None)
        if not callable(exporter):
            return None, "docling_no_export_to_markdown"
        markdown = exporter()
        if not markdown:
            return None, None
        markdown = markdown.strip()
        return (markdown if markdown else None), None

    except Exception as exc:  # noqa: BLE001
        return None, f"docling_failed: {exc}"
    finally:
        if tmp_path:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
