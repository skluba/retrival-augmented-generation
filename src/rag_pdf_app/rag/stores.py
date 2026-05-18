"""Dual indexing: LangChain FAISS (local) + Qdrant (remote)."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from rag_pdf_app.config import Settings


def faiss_dir(settings: Settings) -> Path:
    return Path(settings.faiss_store_path)


def save_faiss_index(store: FAISS, settings: Settings) -> Path:
    out = faiss_dir(settings)
    out.mkdir(parents=True, exist_ok=True)
    store.save_local(str(out))
    return out


def load_faiss_index(embeddings: Embeddings, settings: Settings) -> FAISS:
    folder = faiss_dir(settings)
    return FAISS.load_local(
        str(folder),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def qdrant_client(settings: Settings) -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url)


def reset_qdrant_collection(client: QdrantClient, collection: str, vector_size: int) -> None:
    client.recreate_collection(
        collection_name=collection,
        vectors_config=qmodels.VectorParams(size=vector_size, distance=qmodels.Distance.COSINE),
    )


def stable_point_uuid(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"rag:{chunk_id}"))


def upsert_chunks_qdrant(
    client: QdrantClient,
    *,
    collection: str,
    vectors: list[list[float]],
    texts: list[str],
    metadatas: list[dict[str, Any]],
    batch_size: int = 64,
) -> None:
    if len(vectors) != len(texts) or len(texts) != len(metadatas):
        raise ValueError("vectors, texts, and metadatas length mismatch")

    points: list[qmodels.PointStruct] = []
    for vec, text, meta in zip(vectors, texts, metadatas, strict=True):
        cid = str(meta["chunk_id"])
        payload = {**meta, "text": text}
        points.append(
            qmodels.PointStruct(
                id=stable_point_uuid(cid),
                vector=vec,
                payload=payload,
            )
        )

    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        client.upsert(collection_name=collection, points=batch)
