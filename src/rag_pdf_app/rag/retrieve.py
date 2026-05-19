"""Vector retrieval from FAISS and Qdrant with simple latency comparison."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from qdrant_client import QdrantClient

from rag_pdf_app.config import Settings
from rag_pdf_app.rag.hybrid_fusion import (
    env_page_window_to_zero_based,
    hit_overlaps_page_window,
    intersect_page_windows,
)
from rag_pdf_app.rag.models import RetrievalHit
from rag_pdf_app.rag.query_page_window import inline_window_to_zero_based


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


def _effective_faiss_k(store: FAISS, k: int) -> int:
    """Clamp requested neighbour count to the number of indexed vectors."""

    if k < 1:
        return 1
    idx = getattr(store, "index", None)
    ntotal = getattr(idx, "ntotal", None)
    if isinstance(ntotal, int) and ntotal > 0:
        return min(k, ntotal)
    return k


def faiss_retrieved_chunk_texts(dual: DualRetrievalResult) -> list[str]:
    """Texts from FAISS hits in retrieval order (matches Phase 1 context ordering)."""

    return [h.text for h in dual.faiss_hits]


def _faiss_hits(store: FAISS, query: str, k: int) -> tuple[list[RetrievalHit], float]:
    t0 = time.perf_counter()
    k_eff = _effective_faiss_k(store, k)
    pairs = store.similarity_search_with_score(query, k=k_eff)
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


def faiss_similarity_hits(store: FAISS, query: str, k: int) -> tuple[list[RetrievalHit], float]:
    """Dense top-k retrieval with latency (Phase 1 baseline + Phase 3 hybrid dense leg)."""

    return _faiss_hits(store, query, k)


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


def _apply_page_filters(
    hits: list[RetrievalHit],
    *,
    settings: Settings | None,
    inline_page_window_1based: tuple[int, int] | None,
    top_k: int,
) -> list[RetrievalHit]:
    env_win = (
        env_page_window_to_zero_based(settings.rag_page_filter_min, settings.rag_page_filter_max)
        if settings is not None
        else None
    )
    inline_z = inline_window_to_zero_based(inline_page_window_1based)
    win = intersect_page_windows(env_win, inline_z)
    if win is None:
        return hits[:top_k]
    filt = [h for h in hits if hit_overlaps_page_window(h, win)]
    return (filt or hits)[:top_k]


def _dense_pool_k(
    top_k: int,
    settings: Settings | None,
    inline_page_window_1based: tuple[int, int] | None,
) -> int:
    filtered = False
    if inline_page_window_1based is not None:
        filtered = True
    if settings is not None and (
        settings.rag_page_filter_min is not None or settings.rag_page_filter_max is not None
    ):
        filtered = True
    if not filtered:
        return top_k
    return min(max(top_k * 5, top_k), 50)


def retrieve_dual(
    *,
    query: str,
    embeddings: Embeddings,
    faiss_store: FAISS,
    qdrant: QdrantClient,
    collection: str,
    top_k: int,
    settings: Settings | None = None,
    inline_page_window_1based: tuple[int, int] | None = None,
) -> DualRetrievalResult:
    """Run dense retrieval on both backends.

    When ``settings.rag_hybrid_enabled`` is true, the FAISS leg becomes BM25+dense hybrid with RRF
    and optional cross-encoder / metadata boosts (Phase 3). Qdrant stays a dense baseline for
    comparison. Inline ``pages X-Y`` clauses should be stripped from ``query`` by the caller while
    passing the parsed window via ``inline_page_window_1based``.
    """

    qvec = embeddings.embed_query(query)
    qdrant_note = (
        "Qdrant scores are cosine similarity when collection uses Distance.COSINE "
        "(higher is better)."
    )

    if settings is not None and settings.rag_hybrid_enabled:
        from rag_pdf_app.rag.hybrid_retrieve import hybrid_faiss_retrieval

        fh, f_ms, h_notes = hybrid_faiss_retrieval(
            store=faiss_store,
            query=query,
            settings=settings,
            inline_page_window_1based=inline_page_window_1based,
        )
        notes = [
            *h_notes,
            "FAISS leg uses Phase 3 hybrid ordering for Gemini context.",
            qdrant_note,
        ]
        faiss_metric = "RRF hybrid score (higher is better)"
    else:
        pool_k = _dense_pool_k(top_k, settings, inline_page_window_1based)
        fh, f_ms = _faiss_hits(faiss_store, query, pool_k)
        fh = _apply_page_filters(
            fh,
            settings=settings,
            inline_page_window_1based=inline_page_window_1based,
            top_k=top_k,
        )
        faiss_metric = "L2 distance"
        notes = [
            "FAISS scores are L2 distance for default LangChain FAISS inner product setup "
            "(lower is better when IndexFlatL2).",
            qdrant_note,
        ]

    qh, q_ms = _qdrant_hits(qdrant, collection, qvec, top_k)
    return DualRetrievalResult(
        faiss_hits=fh,
        qdrant_hits=qh,
        faiss_timing=BackendTiming(latency_ms=f_ms, metric=faiss_metric),
        qdrant_timing=BackendTiming(latency_ms=q_ms, metric="cosine similarity"),
        notes=notes,
    )
