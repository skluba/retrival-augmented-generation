import numpy as np
import pytest

from rag_pdf_app.phase6.late_interaction import maxsim_score, reshape_to_slots
from rag_pdf_app.phase6.models import PatchPlacement


def test_reshape_requires_divisibility():
    with pytest.raises(ValueError):
        reshape_to_slots(np.ones(10, dtype=float), slots=8)


def test_maxsim_positive_on_matching_slots():
    q = reshape_to_slots(np.array([1.0, 0.0, 0.0, 2.0], dtype=float), slots=2)
    d = q.copy()
    score = maxsim_score(q, d)
    assert score >= 2.0 - 1e-6


def test_patch_placement_from_payload_compat():
    pay = {
        "page_index": 3,
        "row_index": 1,
        "col_index": 2,
        "x0_px": 40,
        "y0_px": 80,
        "width_px": 256,
        "height_px": 256,
    }
    plc = PatchPlacement(
        page_index=int(pay["page_index"]),
        row_index=int(pay["row_index"]),
        col_index=int(pay["col_index"]),
        x0_px=int(pay["x0_px"]),
        y0_px=int(pay["y0_px"]),
        width_px=int(pay["width_px"]),
        height_px=int(pay["height_px"]),
    )
    assert plc.page_index == 3
    assert plc.width_px == 256
