"""FAISS neighbour count safety (Phase 3 hybrid dense leg uses same path)."""

from __future__ import annotations

from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import FakeEmbeddings

from rag_pdf_app.rag.retrieve import faiss_similarity_hits


def test_faiss_similarity_clamps_k_to_ntotal() -> None:
    texts = ["alpha bravo", "charlie delta"]
    store = FAISS.from_texts(
        texts,
        FakeEmbeddings(size=8),
        metadatas=[
            {"chunk_id": "c1", "page_start": 0},
            {"chunk_id": "c2", "page_start": 1},
        ],
    )
    hits, _ms = faiss_similarity_hits(store, "alpha", k=500)
    assert len(hits) == 2
