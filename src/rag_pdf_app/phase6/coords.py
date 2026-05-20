"""Map patch pixmap rectangles back into PDF user space for cropping."""

from __future__ import annotations

from collections.abc import Mapping

import fitz

from rag_pdf_app.phase6.models import PatchPlacement


def patch_placement_from_payload(payload: Mapping[str, object]) -> PatchPlacement:
    """Rebuild placement from Qdrant payload (:mod:`visual_store`)."""

    return PatchPlacement(
        page_index=int(payload["page_index"]),
        row_index=int(payload["row_index"]),
        col_index=int(payload["col_index"]),
        x0_px=int(payload["x0_px"]),
        y0_px=int(payload["y0_px"]),
        width_px=int(payload["width_px"]),
        height_px=int(payload["height_px"]),
    )


def pdf_rect_from_patch_placement(
    page: fitz.Page,
    pixmap_page_width: int,
    pixmap_page_height: int,
    placement: PatchPlacement,
) -> fitz.Rect:
    """Approximate pixmap patch bounds as a PDF clipping rectangle."""

    prect = page.rect
    sx = placement.x0_px / max(pixmap_page_width, 1) * prect.width
    sy_top = placement.y0_px / max(pixmap_page_height, 1) * prect.height
    sw = placement.width_px / max(pixmap_page_width, 1) * prect.width
    sh = placement.height_px / max(pixmap_page_height, 1) * prect.height
    return fitz.Rect(
        prect.x0 + sx,
        prect.y0 + sy_top,
        prect.x0 + sx + sw,
        prect.y0 + sy_top + sh,
    )


def rerender_patch_png(
    pdf_bytes: bytes,
    *,
    patch_page_index: int,
    placement: PatchPlacement,
    pixmap_page_width: int,
    pixmap_page_height: int,
    dpi: float,
) -> bytes:
    """Re-sample a precise patch using vector PDF geometry."""

    scale = dpi / 72.0
    mat = fitz.Matrix(scale, scale)
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        page = doc.load_page(patch_page_index)
        clip = pdf_rect_from_patch_placement(page, pixmap_page_width, pixmap_page_height, placement)
        pixmap = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
    finally:
        doc.close()
    return pixmap.tobytes(output="png")
