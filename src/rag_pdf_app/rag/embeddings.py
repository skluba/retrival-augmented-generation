"""Vertex AI embeddings via LangChain."""

from __future__ import annotations

from langchain_google_vertexai import VertexAIEmbeddings

from rag_pdf_app.config import Settings


def vertex_text_embeddings(settings: Settings) -> VertexAIEmbeddings:
    return VertexAIEmbeddings(
        project=settings.google_cloud_project,
        location=settings.google_cloud_location,
        model_name=settings.vertex_embedding_model,
    )


def _rough_vertex_embedding_tokens(text: str) -> int:
    """Conservative heuristic so batches stay under the Vertex embed token cap."""

    # Overestimate tokens per string vs naive ratios so greedy batches stay under the API cap.
    return max(1, int(len(text) / 2.0) + 1)


def vertex_embed_documents_batched(
    embedder: VertexAIEmbeddings,
    texts: list[str],
    *,
    batch_size: int,
    max_input_tokens_per_request: int,
) -> list[list[float]]:
    """Pack texts into RPC batches under Vertex limits: ``batch_size`` and total input tokens."""

    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if max_input_tokens_per_request < 1:
        raise ValueError("max_input_tokens_per_request must be >= 1")
    if not texts:
        return []
    out: list[list[float]] = []
    i = 0
    n = len(texts)
    while i < n:
        batch: list[str] = []
        batch_tokens = 0
        while i < n and len(batch) < batch_size:
            t = texts[i]
            t_tokens = _rough_vertex_embedding_tokens(t)
            if t_tokens > max_input_tokens_per_request:
                raise ValueError(
                    "One chunk is larger than the embedding request token budget. "
                    "Reduce RAG_CHUNK_SIZE (or increase RAG_EMBEDDING_MAX_INPUT_TOKENS if the "
                    "model limit allows)."
                )
            if batch_tokens + t_tokens > max_input_tokens_per_request:
                break
            batch.append(t)
            batch_tokens += t_tokens
            i += 1
        out.extend(embedder.embed_documents(batch))
    return out
