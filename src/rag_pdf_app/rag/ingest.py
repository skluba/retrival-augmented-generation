"""Embed chunked text and persist to FAISS + Qdrant."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from langchain_community.vectorstores import FAISS

from rag_pdf_app.config import Settings
from rag_pdf_app.parsing.pipeline import parse_pdf_bytes
from rag_pdf_app.rag.chunking import build_chunks_from_pdf_bytes, chunks_to_langchain_payload
from rag_pdf_app.rag.embeddings import vertex_embed_documents_batched, vertex_text_embeddings
from rag_pdf_app.rag.image_chunks import image_text_chunks_from_parsed_pdf
from rag_pdf_app.rag.stores import (
    qdrant_client,
    reset_qdrant_collection,
    save_faiss_index,
    upsert_chunks_qdrant,
)
from rag_pdf_app.rag.table_chunks import (
    merge_narrative_and_table_chunks,
    table_text_chunks_from_parsed_pdf,
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

    need_structured = settings.rag_table_indexing_enabled or settings.rag_image_indexing_enabled
    parsed = None
    if need_structured:
        embed_b64 = settings.rag_image_indexing_enabled
        img_caption = (
            settings.rag_image_indexing_enabled and settings.rag_ingest_gemini_image_captions
        )
        table_sum = (
            settings.rag_table_indexing_enabled and settings.rag_ingest_gemini_table_summaries
        )
        parsed = parse_pdf_bytes(
            pdf_bytes,
            filename,
            settings=settings,
            embed_image_base64=embed_b64,
            gemini_image_captions=img_caption,
            gemini_table_summaries=table_sum,
            run_docling=settings.rag_ingest_run_docling,
            run_camelot=settings.rag_ingest_run_camelot,
        )
        notes.extend(f"parse_note:{n}" for n in parsed.parsing_notes)

    if settings.rag_table_indexing_enabled and parsed is not None:
        tchunks = table_text_chunks_from_parsed_pdf(parsed)
        chunks, merge_mode = merge_narrative_and_table_chunks(chunks, tchunks)
        notes.append(f"table_index_merge:{merge_mode}")
        notes.append(f"table_chunk_count:{len(tchunks)}")

    if settings.rag_image_indexing_enabled and parsed is not None:
        ichunks = image_text_chunks_from_parsed_pdf(
            parsed,
            min_area_px=settings.rag_image_index_min_area_px,
        )
        chunks, img_merge = merge_narrative_and_table_chunks(chunks, ichunks)
        notes.append(f"image_index_merge:{img_merge}")
        notes.append(f"image_chunk_count:{len(ichunks)}")
        notes.append(f"image_raster_candidates:{len(parsed.images)}")

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
