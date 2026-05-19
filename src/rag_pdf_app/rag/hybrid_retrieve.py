"""Phase 3 hybrid retrieval: dense FAISS leg + BM25 fused via RRF, optional re-ranking."""

from __future__ import annotations

import logging
import time

from langchain_community.vectorstores import FAISS

from rag_pdf_app.config import Settings
from rag_pdf_app.rag.hybrid_fusion import (
    env_page_window_to_zero_based,
    hit_overlaps_page_window,
    intersect_page_windows,
    reciprocal_rank_fusion,
)
from rag_pdf_app.rag.models import RetrievalHit
from rag_pdf_app.rag.query_page_window import inline_window_to_zero_based
from rag_pdf_app.rag.rerank_phase3 import cross_encoder_rerank, metadata_boost_rerank
from rag_pdf_app.rag.retrieve import faiss_similarity_hits
from rag_pdf_app.rag.sparse_bm25 import (
    faiss_snapshot_cache_key,
    find_chunk_by_id,
    get_cached_bm25_index,
)

_LOG = logging.getLogger(__name__)


def _hit_for_chunk(
    store: FAISS,
    chunk_id: str,
    dense_by_id: dict[str, RetrievalHit],
    rrf_score: float,
) -> RetrievalHit | None:
    hit = dense_by_id.get(chunk_id)
    if hit is not None:
        return RetrievalHit(
            chunk_id=hit.chunk_id,
            text=hit.text,
            score=rrf_score,
            metadata=dict(hit.metadata),
        )
    found = find_chunk_by_id(store, chunk_id)
    if found is None:
        return None
    text, meta = found
    return RetrievalHit(chunk_id=chunk_id, text=text, score=rrf_score, metadata=dict(meta))


def _filter_or_fallback(
    merged_ids: list[str],
    *,
    store: FAISS,
    dense_by_id: dict[str, RetrievalHit],
    rrf_scores: dict[str, float],
    page_win: tuple[int, int] | None,
    top_k: int,
    notes: list[str],
) -> list[RetrievalHit]:
    merged_hits: list[RetrievalHit] = []
    for cid in merged_ids:
        hit = _hit_for_chunk(store, cid, dense_by_id, rrf_scores[cid])
        if hit is None:
            continue
        if page_win is None or hit_overlaps_page_window(hit, page_win):
            merged_hits.append(hit)

    if not merged_hits:
        notes.append("hybrid_page_filter_exhausted_fallback_unfiltered")
        for cid in merged_ids:
            hit = _hit_for_chunk(store, cid, dense_by_id, rrf_scores[cid])
            if hit is not None:
                merged_hits.append(hit)
            if len(merged_hits) >= top_k * 3:
                break
    return merged_hits


def hybrid_faiss_retrieval(
    *,
    store: FAISS,
    query: str,
    settings: Settings,
    inline_page_window_1based: tuple[int, int] | None,
) -> tuple[list[RetrievalHit], float, list[str]]:
    """Return (faiss_context_hits, latency_ms, notes)."""

    notes: list[str] = ["phase3_hybrid_dense_plus_bm25_rrf"]
    t0 = time.perf_counter()

    cache_key = faiss_snapshot_cache_key(settings.faiss_store_path)
    bm25, build_dt, cached = get_cached_bm25_index(store, cache_key)
    notes.append(f"bm25_cache_hit:{cached}")
    if build_dt > 0:
        notes.append(f"bm25_build_ms:{build_dt * 1000.0:.1f}")

    dense_pool = settings.rag_hybrid_dense_pool
    sparse_pool = settings.rag_hybrid_sparse_pool

    dense_hits, dense_ms = faiss_similarity_hits(store, query, dense_pool)
    dense_by_id = {h.chunk_id: h for h in dense_hits}

    env_win = env_page_window_to_zero_based(
        settings.rag_page_filter_min,
        settings.rag_page_filter_max,
    )
    inline_z = inline_window_to_zero_based(inline_page_window_1based)
    page_win = intersect_page_windows(env_win, inline_z)

    top_k = settings.rag_top_k

    if bm25 is None:
        notes.append("bm25_unavailable_fallback_dense_only")
        trimmed = [
            h for h in dense_hits if page_win is None or hit_overlaps_page_window(h, page_win)
        ]
        if not trimmed:
            trimmed = dense_hits
        dt_ms = (time.perf_counter() - t0) * 1000.0
        notes.append(f"dense_leg_ms:{dense_ms:.1f}")
        return trimmed[:top_k], dt_ms, notes

    sparse_ranked = bm25.top_chunk_scores(query, sparse_pool)
    sparse_ids = [cid for cid, _ in sparse_ranked]
    dense_ids = [h.chunk_id for h in dense_hits]
    rrf_scores = reciprocal_rank_fusion(
        [dense_ids, sparse_ids],
        rrf_k=settings.rag_rrf_k,
        weights=[settings.rag_rrf_dense_weight, settings.rag_rrf_sparse_weight],
    )
    merged_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)

    merged_hits = _filter_or_fallback(
        merged_ids,
        store=store,
        dense_by_id=dense_by_id,
        rrf_scores=rrf_scores,
        page_win=page_win,
        top_k=top_k,
        notes=notes,
    )

    rerank_cap = max(top_k, settings.rag_cross_encoder_top_n)
    pool = merged_hits[:rerank_cap]

    if settings.rag_metadata_boost_enabled and pool:
        pool = metadata_boost_rerank(pool, query)
        notes.append("metadata_boost_applied")

    if settings.rag_cross_encoder_model:
        try:
            pool = cross_encoder_rerank(query, pool, model_name=settings.rag_cross_encoder_model)
            notes.append("cross_encoder_applied")
        except RuntimeError as exc:
            notes.append(f"cross_encoder_skipped:{exc}")
            _LOG.warning("cross-encoder rerank skipped: %s", exc)

    final = pool[:top_k]
    dt_ms = (time.perf_counter() - t0) * 1000.0
    notes.append(f"dense_leg_ms:{dense_ms:.1f}")
    return final, dt_ms, notes
