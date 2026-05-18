"""Dual indexing: LangChain FAISS (local) + Qdrant (remote)."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import dependable_faiss_import
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from rag_pdf_app.config import Settings

_FAISS_MANIFEST = "docstore.manifest.json"
_LC_INDEX_NAME = "index"


def faiss_dir(settings: Settings) -> Path:
    return Path(settings.faiss_store_path)


def save_faiss_index(store: FAISS, settings: Settings) -> Path:
    """Persist FAISS index + JSON docstore (no pickle — avoids unsafe deserialization)."""

    out = faiss_dir(settings).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    if not isinstance(store.docstore, InMemoryDocstore):
        raise TypeError("Safe FAISS export requires an InMemoryDocstore.")

    faiss_mod = dependable_faiss_import()
    faiss_mod.write_index(store.index, str(out / f"{_LC_INDEX_NAME}.faiss"))

    docs_payload: dict[str, dict[str, Any]] = {}
    for did, doc in store.docstore._dict.items():
        docs_payload[did] = {
            "page_content": doc.page_content,
            "metadata": dict(doc.metadata),
        }
    manifest: dict[str, Any] = {
        "format": "rag_pdf_app_faiss_manifest_v1",
        "distance_strategy": store.distance_strategy.value,
        "normalize_L2": store._normalize_L2,
        "index_to_docstore_id": {str(k): v for k, v in store.index_to_docstore_id.items()},
        "documents": docs_payload,
    }
    manifest_path = out / _FAISS_MANIFEST
    tmp_path = out / f"{_FAISS_MANIFEST}.tmp"
    tmp_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    tmp_path.replace(manifest_path)

    legacy_pkl = out / f"{_LC_INDEX_NAME}.pkl"
    if legacy_pkl.is_file():
        legacy_pkl.unlink()

    return out


def load_faiss_index(embeddings: Embeddings, settings: Settings) -> FAISS:
    """Load index written by ``save_faiss_index`` (manifest + ``.faiss`` only)."""

    folder = faiss_dir(settings).expanduser().resolve()
    manifest_path = folder / _FAISS_MANIFEST
    if not manifest_path.is_file():
        legacy_pkl = folder / f"{_LC_INDEX_NAME}.pkl"
        if legacy_pkl.is_file():
            raise RuntimeError(
                "This FAISS directory uses a legacy pickle docstore. "
                "Run **Ingest & index** again to rebuild with the pickle-free format."
            )
        raise FileNotFoundError(
            f"No FAISS manifest at {manifest_path}; ingest or check FAISS_STORE_PATH."
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "rag_pdf_app_faiss_manifest_v1":
        raise ValueError(f"Unsupported FAISS manifest: {manifest.get('format')!r}")

    faiss_mod = dependable_faiss_import()
    index = faiss_mod.read_index(str(folder / f"{_LC_INDEX_NAME}.faiss"))

    documents: dict[str, Document] = {}
    for did, payload in manifest["documents"].items():
        meta = payload.get("metadata") or {}
        documents[did] = Document(
            page_content=str(payload.get("page_content", "")),
            metadata=dict(meta) if isinstance(meta, dict) else {},
        )
    docstore = InMemoryDocstore(documents)
    index_to_docstore_id = {int(k): str(v) for k, v in manifest["index_to_docstore_id"].items()}

    return FAISS(
        embeddings,
        index,
        docstore,
        index_to_docstore_id,
        normalize_L2=bool(manifest.get("normalize_L2", False)),
        distance_strategy=DistanceStrategy(manifest["distance_strategy"]),
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
