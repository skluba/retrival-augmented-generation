"""Multi-hop FAISS merge behaviour (offline)."""

from __future__ import annotations

from rag_pdf_app.rag.models import RetrievalHit
from rag_pdf_app.rag.multi_hop import merge_faiss_hits_deduped


def test_merge_faiss_hits_dedupes_by_chunk_id_preserving_order() -> None:
    h1 = RetrievalHit(
        chunk_id="c1",
        text="one",
        score=1.0,
        metadata={"page_start": 0},
    )
    h2 = RetrievalHit(
        chunk_id="c2",
        text="two",
        score=0.9,
        metadata={"page_start": 1},
    )
    h1_dup = RetrievalHit(
        chunk_id="c1",
        text="dup",
        score=0.5,
        metadata={"page_start": 0},
    )
    h3 = RetrievalHit(chunk_id="c3", text="three", score=0.8, metadata={})
    merged = merge_faiss_hits_deduped([h1, h2], [h1_dup, h3], top_k=3)
    assert [h.chunk_id for h in merged] == ["c1", "c2", "c3"]


def test_merge_respects_top_k() -> None:
    a = [RetrievalHit(chunk_id=f"c{i}", text=str(i), score=1.0, metadata={}) for i in range(5)]
    b = [RetrievalHit(chunk_id=f"d{i}", text=str(i), score=1.0, metadata={}) for i in range(5)]
    merged = merge_faiss_hits_deduped(a, b, top_k=4)
    assert len(merged) == 4
