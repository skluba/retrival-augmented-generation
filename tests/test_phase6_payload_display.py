"""Tests for Qdrant payload coercion used by the Phase 6 Streamlit pane."""

from __future__ import annotations

import pytest

from rag_pdf_app.phase6.payload_display import (
    sanitized_phase6_patch_display,
    strict_float_metric,
    strict_uint_field,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (3, 3),
        ("x", 0),
        (True, 0),
        (False, 0),
        (3.1, 0),
        (3.0, 3),
        (-2, 0),
        (600_001, 0),
        (None, 0),
    ],
)
def test_strict_uint_field(raw: object, expected: int) -> None:
    assert strict_uint_field(raw, fallback=0, cap=500_000) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (0.42, 0.42),
        ("nan", 0.0),
        ("inf", 0.0),
        (float("nan"), 0.0),
        ("3.14", 3.14),
        ("", 0.0),
        (None, 0.0),
        (True, 0.0),
        (10**12, 1e9),
    ],
)
def test_strict_float_metric(raw: object, expected: float) -> None:
    assert strict_float_metric(raw) == pytest.approx(expected, rel=0, abs=1e-12)


def test_sanitized_patch_rejects_poison_geometry() -> None:
    sane = sanitized_phase6_patch_display(
        {
            "page_index": "1evil",
            "row_index": "[link](evil)",
            "col_index": 2,
            "x0_px": 0,
            "y0_px": "y",
            "width_px": 0,
            "height_px": -5,
            "pixmap_page_width": 100,
            "pixmap_page_height": 200,
        }
    )
    assert sane.placement.page_index == 0
    assert sane.placement.row_index == 0
    assert sane.placement.col_index == 2
    assert sane.placement.y0_px == 0
    assert sane.placement.width_px == 1
    assert sane.placement.height_px == 1
    assert sane.pixmap_page_width == 100
