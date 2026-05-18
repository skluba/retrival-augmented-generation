"""Embed chunked text and persist to FAISS + Qdrant."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from langchain_community.vectorstores import FAISS

from rag_pdf_app.config import Settings
from rag_pdf_app.rag.chunking import build_chunks_from_pdf_bytes, chunks_to_langchain_payload
from rag_pdf_app.rag.embeddings import vertex_embed_documents_batched, vertex_text_embeddings
from rag_pdf_app.rag.stores import (
    qdrant_client,
    reset_qdrant_collection,
    save_faiss_index,
    upsert_chunks_qdrant,
)


@dataclass
class IngestOutcome:
    chunk_count: int
    embedding_dimensions: int
    pdf_sha256: str
    faiss_path: str
    qdrant_collection: str
    notes: list[str] = field(default_factory=list)


def ingest_pdf_bytes_to_indexes(
    settings: Settings,
    pdf_bytes: bytes,
    filename: str,
    *,
    use_layout_chunking: bool = True,
) -> IngestOutcome:
    """Chunk PDF text, embed with Vertex, write FAISS dir + Qdrant collection."""

    sha = hashlib.sha256(pdf_bytes).hexdigest()
    notes: list[str] = []
    chunks, cnotes = build_chunks_from_pdf_bytes(
        pdf_bytes,
        filename=filename,
        chunk_size=settings.rag_chunk_size,
        chunk_overlap=settings.rag_chunk_overlap,
        use_layout=use_layout_chunking,
    )
    notes.extend(cnotes)
    if not chunks:
        raise ValueError("No text extracted from PDF — cannot index.")

    texts, metadatas = chunks_to_langchain_payload(chunks)
    embedder = vertex_text_embeddings(settings)
    vectors = vertex_embed_documents_batched(
        embedder,
        texts,
        batch_size=settings.rag_embedding_batch_size,
        max_input_tokens_per_request=settings.rag_embedding_max_input_tokens,
    )
    if not vectors:
        raise RuntimeError("Embedding provider returned no vectors.")
    dim = len(vectors[0])

    store = FAISS.from_embeddings(
        list(zip(texts, vectors, strict=True)),
        embedder,
        metadatas=metadatas,
    )
    out_dir = save_faiss_index(store, settings)

    client = qdrant_client(settings)
    reset_qdrant_collection(client, settings.rag_qdrant_collection, dim)
    upsert_chunks_qdrant(
        client,
        collection=settings.rag_qdrant_collection,
        vectors=vectors,
        texts=texts,
        metadatas=metadatas,
    )

    notes.append(f"indexed_pdf_sha256:{sha}")
    return IngestOutcome(
        chunk_count=len(chunks),
        embedding_dimensions=dim,
        pdf_sha256=sha,
        faiss_path=str(out_dir.resolve()),
        qdrant_collection=settings.rag_qdrant_collection,
        notes=notes,
    )


def ingest_pdf_path(
    settings: Settings,
    pdf_path: str | Path,
    *,
    use_layout_chunking: bool = True,
) -> IngestOutcome:
    path = Path(pdf_path)
    data = path.read_bytes()
    return ingest_pdf_bytes_to_indexes(
        settings, data, path.name, use_layout_chunking=use_layout_chunking
    )
