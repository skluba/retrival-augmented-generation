from rag_pdf_app.phase6.grid import count_patches_est, iter_patch_bounds, patch_rectangles


def test_iter_patch_bounds_non_overlapping_grid():
    rows = list(iter_patch_bounds(300, 200, patch_size_px=128, stride_px=128))
    assert rows
    for _r, _c, x0, y0, w, h in rows:
        assert w <= 128 and h <= 128
        assert 0 <= x0 < 300 and 0 <= y0 < 200


def test_patch_rectangle_count_matches_helper():
    w, h, ps, st = 500, 500, 256, 192
    rects = patch_rectangles(w, h, ps, st)
    assert count_patches_est(w, h, ps, st) == len(rects)
