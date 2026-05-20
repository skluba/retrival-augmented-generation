"""Patch tiling over a page pixmap (pure geometry for tests)."""

from __future__ import annotations


def iter_patch_bounds(
    page_width_px: int,
    page_height_px: int,
    patch_size_px: int,
    stride_px: int,
):
    """Yield ``(row, col, x0, y0, width, height)`` rects with clipped edge tiles."""

    if patch_size_px < 64 or stride_px < 16:
        msg = "patch_size_px and stride_px must be reasonable"
        raise ValueError(msg)
    if stride_px > patch_size_px:
        msg = "stride_px larger than patch_size_px would skip content"
        raise ValueError(msg)

    row_idx = -1
    y = 0
    while y < page_height_px:
        row_idx += 1
        col_idx = -1
        height = min(patch_size_px, page_height_px - y)
        x = 0
        while x < page_width_px:
            col_idx += 1
            width = min(patch_size_px, page_width_px - x)
            yield row_idx, col_idx, x, y, width, height
            x += stride_px
        y += stride_px


def patch_rectangles(
    page_width_px: int,
    page_height_px: int,
    patch_size_px: int,
    stride_px: int,
) -> list[tuple[int, int, int, int]]:
    """Flattened ``(x0, y0, width, height)`` for compatibility."""

    out: list[tuple[int, int, int, int]] = []
    for _r, _c, x0, y0, w, h in iter_patch_bounds(
        page_width_px, page_height_px, patch_size_px, stride_px
    ):
        out.append((x0, y0, w, h))
    return out


def count_patches_est(page_width_px: int, page_height_px: int, patch_size: int, stride: int) -> int:
    """Fast patch count aligned with :func:`patch_rectangles`."""

    return len(patch_rectangles(page_width_px, page_height_px, patch_size, stride))
