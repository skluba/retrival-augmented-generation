"""Batching for Vertex embedding RPC limits."""

from unittest.mock import MagicMock

from rag_pdf_app.rag.embeddings import vertex_embed_documents_batched


def test_vertex_embed_documents_batched_splits_calls() -> None:
    embedder = MagicMock()

    def fake_embed(batch: list[str]) -> list[list[float]]:
        return [[float(i)] for i in range(len(batch))]

    embedder.embed_documents.side_effect = fake_embed
    texts = [str(i) for i in range(5)]
    out = vertex_embed_documents_batched(
        embedder,
        texts,
        batch_size=2,
        max_input_tokens_per_request=100_000,
    )
    assert len(out) == 5
    assert embedder.embed_documents.call_count == 3


def test_vertex_embed_documents_batched_splits_on_token_budget() -> None:
    embedder = MagicMock()
    embedder.embed_documents.side_effect = lambda b: [[0.0, 0.0] for _ in b]
    chunk = "x" * 3000
    texts = [chunk, chunk, chunk]
    vertex_embed_documents_batched(
        embedder,
        texts,
        batch_size=250,
        max_input_tokens_per_request=2000,
    )
    assert embedder.embed_documents.call_count == 3
