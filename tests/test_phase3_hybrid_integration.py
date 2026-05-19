"""Integration-style checks for Phase 3 FAISS hybrid retrieval (BM25 + dense + RRF)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import FakeEmbeddings

from rag_pdf_app.config import Settings
from rag_pdf_app.rag.hybrid_retrieve import hybrid_faiss_retrieval
from rag_pdf_app.rag.retrieve import retrieve_dual
from rag_pdf_app.rag.sparse_bm25 import clear_bm25_cache


@pytest.fixture(autouse=True)
def _clear_bm25_cache(request: pytest.FixtureRequest) -> None:
    clear_bm25_cache()
    request.addfinalizer(clear_bm25_cache)


def _pad_metas(count: int, *, start_page: int = 100) -> tuple[list[str], list[dict[str, object]]]:
    """Filler chunks so FAISS ``k`` (min 5 from Settings) never exceeds ``ntotal``."""

    texts = [f"Filler passage index {i} about miscellaneous logistics." for i in range(count)]
    metas = [
        {
            "chunk_id": f"c_pad_{start_page + i}",
            "page_start": start_page + i,
            "page_end": start_page + i,
            "content_type": "narrative",
        }
        for i in range(count)
    ]
    return texts, metas


def _settings(tmp_faiss_root: Path, **overrides: object) -> Settings:
    base: dict[str, object] = {
        "google_cloud_project": "test-project-qa",
        "faiss_store_path": str(tmp_faiss_root),
        "rag_hybrid_enabled": True,
        "rag_top_k": 3,
        "rag_hybrid_dense_pool": 5,
        "rag_hybrid_sparse_pool": 5,
        "rag_rrf_k": 60,
        "rag_cross_encoder_model": None,
        "rag_metadata_boost_enabled": True,
    }
    base.update(overrides)
    return Settings(**base)


def _store(texts: list[str], metadatas: list[dict[str, object]]) -> FAISS:
    return FAISS.from_texts(texts, FakeEmbeddings(size=16), metadatas=metadatas)


def test_hybrid_faiss_retrieval_runs_bm25_rrf_and_returns_hits(tmp_path: Path) -> None:
    pad_texts, pad_metas = _pad_metas(2, start_page=50)
    texts = [
        "Disbursement obligations for IFC fiscal year reporting.",
        "Unrelated narrative about parks and recreation seasons.",
        "Another passage discussing ocean currents and wind patterns.",
        *pad_texts,
    ]
    metas = [
        {
            "chunk_id": "c_disburse",
            "page_start": 0,
            "page_end": 0,
            "content_type": "narrative",
            "section_hint": "Disbursement obligations for IFC fiscal year reporting.",
        },
        {"chunk_id": "c_parks", "page_start": 1, "page_end": 1, "content_type": "narrative"},
        {"chunk_id": "c_ocean", "page_start": 2, "page_end": 2, "content_type": "narrative"},
        *pad_metas,
    ]
    store = _store(texts, metas)
    settings = _settings(tmp_path / "faiss_hybrid_rrf", rag_top_k=5)

    hits, _ms, notes = hybrid_faiss_retrieval(
        store=store,
        query="disbursement obligations IFC",
        settings=settings,
        inline_page_window_1based=None,
    )

    joined_notes = " ".join(notes)
    assert "phase3_hybrid_dense_plus_bm25_rrf" in joined_notes
    assert "bm25_unavailable" not in joined_notes
    assert len(hits) <= settings.rag_top_k
    ids = [h.chunk_id for h in hits]
    assert "c_disburse" in ids


def test_hybrid_page_window_keeps_only_overlapping_chunks(tmp_path: Path) -> None:
    pad_texts, pad_metas = _pad_metas(3, start_page=50)
    texts = [
        "Early chapter discusses unrelated boilerplate.",
        "Material IFC disbursement metrics appear in this chapter.",
        *pad_texts,
    ]
    metas = [
        {"chunk_id": "c_early", "page_start": 0, "page_end": 0, "content_type": "narrative"},
        {"chunk_id": "c_late", "page_start": 10, "page_end": 10, "content_type": "narrative"},
        *pad_metas,
    ]
    store = _store(texts, metas)
    settings = _settings(tmp_path / "faiss_hybrid_pages")

    hits, _ms, notes = hybrid_faiss_retrieval(
        store=store,
        query="IFC disbursement metrics chapter",
        settings=settings,
        inline_page_window_1based=(11, 11),
    )

    assert [h.chunk_id for h in hits] == ["c_late"]
    assert not any("hybrid_page_filter_exhausted_fallback_unfiltered" in n for n in notes)


def test_hybrid_falls_back_dense_when_chunk_ids_missing(tmp_path: Path) -> None:
    fillers = [f"Noise passage number {i} without lexical ids." for i in range(3)]
    store = FAISS.from_texts(
        ["alpha beta gamma delta", "epsilon zeta eta theta", *fillers],
        FakeEmbeddings(size=8),
        metadatas=[
            {"page_start": 0},
            {"page_start": 1},
            *[{"page_start": 2 + i} for i in range(3)],
        ],
    )
    settings = _settings(tmp_path / "faiss_hybrid_no_ids")

    hits, _ms, notes = hybrid_faiss_retrieval(
        store=store,
        query="alpha gamma",
        settings=settings,
        inline_page_window_1based=None,
    )

    assert any("bm25_unavailable_fallback_dense_only" in n for n in notes)
    assert len(hits) >= 1


def test_retrieve_dual_hybrid_faiss_leg_with_mock_qdrant(tmp_path: Path) -> None:
    pad_texts, pad_metas = _pad_metas(3, start_page=20)
    texts = ["Topic one about governance.", "Topic two about disbursements and IFC.", *pad_texts]
    metas = [
        {"chunk_id": "c1", "page_start": 0, "page_end": 0, "content_type": "narrative"},
        {"chunk_id": "c2", "page_start": 1, "page_end": 1, "content_type": "narrative"},
        *pad_metas,
    ]
    store = _store(texts, metas)
    settings = _settings(tmp_path / "faiss_dual_hybrid", rag_top_k=2)

    qdrant = MagicMock()
    qdrant.query_points.return_value = MagicMock(points=[])

    dual = retrieve_dual(
        query="disbursements IFC",
        embeddings=FakeEmbeddings(size=16),
        faiss_store=store,
        qdrant=qdrant,
        collection="dummy",
        top_k=settings.rag_top_k,
        settings=settings,
        inline_page_window_1based=None,
    )

    assert dual.faiss_timing.metric == "RRF hybrid score (higher is better)"
    assert dual.faiss_hits
    assert "phase3_hybrid_dense_plus_bm25_rrf" in " ".join(dual.notes)
    qdrant.query_points.assert_called_once()
