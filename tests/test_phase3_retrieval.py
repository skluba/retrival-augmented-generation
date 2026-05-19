"""Offline checks for Phase 3 hybrid retrieval helpers."""

from __future__ import annotations

from rag_pdf_app.rag.chunk_metadata import infer_chunk_content_type
from rag_pdf_app.rag.hybrid_fusion import (
    env_page_window_to_zero_based,
    intersect_page_windows,
    reciprocal_rank_fusion,
)
from rag_pdf_app.rag.query_page_window import strip_inline_page_window


def test_reciprocal_rank_fusion_prefers_shared_top_docs() -> None:
    dense = ["a", "b", "c"]
    sparse = ["b", "c", "d"]
    scores = reciprocal_rank_fusion([dense, sparse], rrf_k=60)
    assert scores["b"] > scores["d"]
    assert scores["c"] > scores["d"]


def test_reciprocal_rank_fusion_weights_boost_lexical_leg() -> None:
    dense = ["x", "y"]
    sparse = ["y", "x"]
    weighted = reciprocal_rank_fusion([dense, sparse], rrf_k=60, weights=[1.0, 3.0])
    assert weighted["y"] > weighted["x"]


def test_strip_inline_page_window() -> None:
    q, win = strip_inline_page_window("Net income pages 4-8 for IFC")
    assert win == (4, 8)
    assert "4-8" not in q


def test_env_page_window_to_zero_based() -> None:
    assert env_page_window_to_zero_based(10, 12) == (9, 11)
    assert env_page_window_to_zero_based(None, 5) == (0, 4)


def test_intersect_page_windows() -> None:
    assert intersect_page_windows((0, 5), (3, 9)) == (3, 5)
    assert intersect_page_windows((0, 2), (5, 6)) is None


def test_infer_chunk_content_type_detects_figure_language() -> None:
    assert infer_chunk_content_type("Refer to Figure 3 for disbursements.", None) == "figure_ref"


def test_infer_chunk_content_type_currency_matches_table_heuristic() -> None:
    """Dollar amounts and ``million`` are classified as table-dense before numeric_heavy."""

    assert infer_chunk_content_type("Total assets were $110,547 million.", None) == "table_dense"
