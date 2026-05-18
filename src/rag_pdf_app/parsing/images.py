"""Raster / vector image extraction via PyMuPDF."""

from __future__ import annotations

import base64

import fitz  # PyMuPDF

from rag_pdf_app.parsing.models import BBox, ImageBlock


def _mime_from_extension(ext: str) -> str:
    ext_l = ext.lower()
    if ext_l in {"jpg", "jpeg"}:
        return "image/jpeg"
    if ext_l:
        return f"image/{ext_l}"
    return "image/png"


def _bbox_for_image_xref(page: fitz.Page, page_index: int, xref: int) -> BBox:
    rects = page.get_image_rects(xref)
    if not rects:
        return BBox(page_index=page_index, x0=0.0, y0=0.0, x1=0.0, y1=0.0)
    r = rects[0]
    return BBox(
        page_index=page_index,
        x0=float(r.x0),
        y0=float(r.y0),
        x1=float(r.x1),
        y1=float(r.y1),
    )


def _try_extract_image_dict(doc: fitz.Document, xref: int) -> dict | None:
    try:
        return doc.extract_image(xref)
    except Exception:  # noqa: BLE001 — PyMuPDF xref/image extractor surface varies widely
        return None


def _image_block_from_extract(
    xref: int,
    page_index: int,
    page: fitz.Page,
    base: dict,
    *,
    embed_base64: bool,
) -> ImageBlock:
    image_bytes: bytes = base["image"]
    ext = str(base.get("ext", "png")).lower()
    mime = _mime_from_extension(ext)
    bbox = _bbox_for_image_xref(page, page_index, xref)
    b64: str | None = None
    if embed_base64:
        b64 = base64.b64encode(image_bytes).decode("ascii")
    return ImageBlock(
        xref=xref,
        page_index=page_index,
        bbox=bbox,
        mime_type=mime,
        width_px=base.get("width"),
        height_px=base.get("height"),
        image_bytes_b64=b64,
        metadata={
            "colorspace": base.get("colorspace"),
            "bpc": base.get("bpc"),
            "extract_image_keys": sorted(str(k) for k in base.keys()),
        },
    )


def extract_images_pymupdf(
    pdf_bytes: bytes,
    *,
    embed_base64: bool = True,
) -> list[ImageBlock]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    blocks: list[ImageBlock] = []

    try:
        for page_index in range(len(doc)):
            page = doc[page_index]
            for img in page.get_images(full=True):
                xref = int(img[0])
                base = _try_extract_image_dict(doc, xref)
                if base is None:
                    continue
                blocks.append(
                    _image_block_from_extract(
                        xref,
                        page_index,
                        page,
                        base,
                        embed_base64=embed_base64,
                    )
                )
    finally:
        doc.close()

    return dedupe_near_identical_images(blocks)


def dedupe_near_identical_images(images: list[ImageBlock]) -> list[ImageBlock]:
    """Drop obvious duplicates that share xref + bbox (covers repeated placements)."""

    seen: set[tuple[int, int, float, float, float, float]] = set()
    out: list[ImageBlock] = []
    for im in sorted(images, key=lambda b: (b.page_index, b.xref, min(b.bbox.y0, b.bbox.y1))):
        key = (im.page_index, im.xref, im.bbox.x0, im.bbox.y0, im.bbox.x1, im.bbox.y1)
        if key in seen:
            continue
        seen.add(key)
        out.append(im)
    return out
