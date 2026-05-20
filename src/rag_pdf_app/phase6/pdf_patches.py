"""Rasterise PDF pages and emit PNG crops for downstream CLIP embeddings."""

from __future__ import annotations

import hashlib
import io
from collections.abc import Iterator

import fitz  # PyMuPDF
from PIL import Image

from rag_pdf_app.phase6.grid import iter_patch_bounds
from rag_pdf_app.phase6.models import PatchPlacement, PatchRecord


def sha256_pdf(pdf_bytes: bytes) -> str:
    return hashlib.sha256(pdf_bytes).hexdigest()


def iter_pdf_patch_records(
    pdf_bytes: bytes,
    *,
    source_filename: str,
    dpi: float,
    patch_size_px: int,
    stride_px: int,
    max_pages: int,
) -> Iterator[PatchRecord]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    pdf_sha256 = sha256_pdf(pdf_bytes)

    matrix = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    limit = max(1, min(max_pages, len(doc)))

    try:
        for page_index in range(limit):
            page = doc.load_page(page_index)
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)

            pim = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)

            pw, ph = pim.size
            for row_index, col_index, x0, y0, w, h in iter_patch_bounds(
                pw, ph, patch_size_px, stride_px
            ):
                cropped = pim.crop((x0, y0, x0 + w, y0 + h))
                buf = io.BytesIO()
                cropped.save(buf, format="PNG", optimize=True)
                png = buf.getvalue()

                pid = hashlib.sha256(
                    f"{pdf_sha256}|p={page_index}|r={row_index}|c={col_index}|dpi={dpi}".encode(),
                ).hexdigest()[:26]
                patch_id = f"p6-{pid}"

                yield PatchRecord(
                    patch_id=patch_id,
                    pdf_sha256=pdf_sha256,
                    source_filename=source_filename,
                    placement=PatchPlacement(
                        page_index=page_index,
                        row_index=row_index,
                        col_index=col_index,
                        x0_px=x0,
                        y0_px=y0,
                        width_px=w,
                        height_px=h,
                    ),
                    patch_png_bytes=png,
                    dpi=int(dpi),
                    pixmap_page_width=pw,
                    pixmap_page_height=ph,
                )
    finally:
        doc.close()
