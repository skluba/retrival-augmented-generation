"""Explicit figure-number gating near embedded rasters (Phase 5.2 noise control)."""

from __future__ import annotations

from rag_pdf_app.parsing.figure_cues import nearby_text_has_explicit_figure_label


def test_detects_common_numbered_labels() -> None:
    assert nearby_text_has_explicit_figure_label("Figure 3: Net income", None)
    assert nearby_text_has_explicit_figure_label(None, "as shown in Fig. 2 ")
    assert nearby_text_has_explicit_figure_label("", " Fig 4 Revenue mix")
    assert nearby_text_has_explicit_figure_label(
        "",
        "capital diagram",
        "Annotated copy of Figure 12 from source data",
    )


def test_rejects_unrelated_section_headings() -> None:
    assert not nearby_text_has_explicit_figure_label(
        "INDEPENDENT AUDITOR’S REPORT",
        "Management responsibilities",
    )


def test_empty_inputs_false() -> None:
    assert not nearby_text_has_explicit_figure_label(None, None, None)
    assert not nearby_text_has_explicit_figure_label("", "   ", None)
