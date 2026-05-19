"""Vector retrieval from FAISS and Qdrant with simple latency comparison."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from qdrant_client import QdrantClient


@dataclass
class RetrievalHit:
    chunk_id: str
    text: str
    score: float
    metadata: dict[str, Any]


@dataclass
class BackendTiming:
    latency_ms: float
    metric: str


@dataclass
class DualRetrievalResult:
    faiss_hits: list[RetrievalHit]
    qdrant_hits: list[RetrievalHit]
    faiss_timing: BackendTiming
    qdrant_timing: BackendTiming
    notes: list[str] = field(default_factory=list)


def faiss_retrieved_chunk_texts(dual: DualRetrievalResult) -> list[str]:
    """Texts from FAISS hits in retrieval order (matches Phase 1 context ordering)."""

    return [h.text for h in dual.faiss_hits]


def _faiss_hits(store: FAISS, query: str, k: int) -> tuple[list[RetrievalHit], float]:
    t0 = time.perf_counter()
    pairs = store.similarity_search_with_score(query, k=k)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    hits: list[RetrievalHit] = []
    for doc, score in pairs:
        meta = dict(doc.metadata)
        cid = str(meta.get("chunk_id", ""))
        hits.append(
            RetrievalHit(
                chunk_id=cid,
                text=doc.page_content,
                score=float(score),
                metadata=meta,
            )
        )
    return hits, dt_ms


def _qdrant_hits(
    client: QdrantClient,
    collection: str,
    query_vector: list[float],
    k: int,
) -> tuple[list[RetrievalHit], float]:
    t0 = time.perf_counter()
    res = client.query_points(
        collection_name=collection,
        query=query_vector,
        limit=k,
        with_payload=True,
    )
    dt_ms = (time.perf_counter() - t0) * 1000.0
    hits: list[RetrievalHit] = []
    for pt in res.points:
        payload = dict(pt.payload or {})
        text = str(payload.pop("text", ""))
        cid = str(payload.get("chunk_id", ""))
        score = float(pt.score) if pt.score is not None else 0.0
        hits.append(
            RetrievalHit(chunk_id=cid, text=text, score=score, metadata=payload),
        )
    return hits, dt_ms


def retrieve_dual(
    *,
    query: str,
    embeddings: Embeddings,
    faiss_store: FAISS,
    qdrant: QdrantClient,
    collection: str,
    top_k: int,
) -> DualRetrievalResult:
    """Run identical dense retrieval against both backends."""

    qvec = embeddings.embed_query(query)
    fh, f_ms = _faiss_hits(faiss_store, query, top_k)
    qh, q_ms = _qdrant_hits(qdrant, collection, qvec, top_k)
    notes = [
        "FAISS scores are L2 distance for default LangChain FAISS inner product setup "
        "(lower is better when IndexFlatL2).",
        "Qdrant scores are cosine similarity when collection uses Distance.COSINE "
        "(higher is better).",
    ]
    return DualRetrievalResult(
        faiss_hits=fh,
        qdrant_hits=qh,
        faiss_timing=BackendTiming(latency_ms=f_ms, metric="L2 distance"),
        qdrant_timing=BackendTiming(latency_ms=q_ms, metric="cosine similarity"),
        notes=notes,
    )
