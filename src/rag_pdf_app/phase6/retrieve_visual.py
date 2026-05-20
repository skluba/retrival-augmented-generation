"""Dense + optional pseudo-MaxSim reranking over multilingual CLIP patch vectors."""

from __future__ import annotations

from typing import Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.models import FieldCondition, Filter, MatchValue

from rag_pdf_app.config import Settings
from rag_pdf_app.phase6.clip_embed import encode_queries_clip
from rag_pdf_app.phase6.late_interaction import maxsim_score, reshape_to_slots
from rag_pdf_app.phase6.models import PatchRetrievalHit


def _dense_list_vector(raw: object | None) -> list[float] | None:
    """Normalise Qdrant vector payloads (dense list vs single-key named-vector dict)."""

    if raw is None:
        return None
    if isinstance(raw, list) and raw and isinstance(raw[0], (int, float)):
        return [float(v) for v in raw]
    if isinstance(raw, dict):
        for v in raw.values():
            mapped = _dense_list_vector(v)
            if mapped is not None:
                return mapped
    return None


def _hits_cosine(points: Any, *, limit: int) -> list[PatchRetrievalHit]:
    hits: list[PatchRetrievalHit] = []
    for pt in points:
        payload = dict(pt.payload or {})
        cid = str(payload.get("patch_id", pt.id))
        score = float(pt.score if pt.score is not None else 0.0)
        hits.append(
            PatchRetrievalHit(patch_id=cid, score=score, payload=payload, coarse_score=score),
        )
    return hits[:limit]


def _maxsim_rerank(
    query_vector: list[float],
    coarse_hits: Any,
    *,
    slots: int,
    top_k: int,
) -> tuple[list[PatchRetrievalHit], str]:
    qs = reshape_to_slots(np.asarray(query_vector, dtype=np.float64), slots)
    ranked: list[tuple[float, PatchRetrievalHit]] = []
    for pt in coarse_hits:
        dv_raw = getattr(pt, "vector", None)
        dv_list = _dense_list_vector(dv_raw)
        if dv_list is None:
            continue
        payload = dict(pt.payload or {})
        cid = str(payload.get("patch_id", pt.id))
        coarse_score = float(pt.score if pt.score is not None else 0.0)
        doc_slots = reshape_to_slots(np.asarray(dv_list, dtype=np.float64), slots)
        rerank_score = maxsim_score(qs, doc_slots)
        ranked.append(
            (
                rerank_score,
                PatchRetrievalHit(
                    patch_id=cid,
                    score=rerank_score,
                    payload=payload,
                    coarse_score=coarse_score,
                ),
            )
        )

    if not ranked:
        return (
            _hits_cosine(coarse_hits, limit=top_k),
            f"maxsim_failed:{slots}",
        )

    ranked.sort(key=lambda item: (-item[0], -(item[1].coarse_score or 0.0)))
    merged = [item[1] for item in ranked]
    tag = f"maxsim_pseudo_slots:{slots}"
    return merged[:top_k], tag


def retrieve_phase6_visual_patches(
    client: QdrantClient,
    settings: Settings,
    *,
    query: str,
    pdf_sha256: str | None = None,
    device: str | None = None,
) -> tuple[list[PatchRetrievalHit], dict[str, str]]:
    """Return ranked patch hits with cosine scores and optional MaxSim-style rerank."""

    query_vector = encode_queries_clip(texts=[query], settings=settings, device=device)[0]

    filt: Filter | None = None
    if pdf_sha256:
        filt = Filter(
            must=[
                FieldCondition(
                    key="pdf_sha256",
                    match=MatchValue(value=str(pdf_sha256)),
                )
            ]
        )

    coarse = settings.phase6_visual_prefetch
    coarse_hits = client.query_points(
        collection_name=settings.phase6_qdrant_collection,
        query=query_vector,
        query_filter=filt,
        limit=coarse,
        with_payload=True,
        with_vectors=True,
    ).points

    telemetry: dict[str, str] = {
        "prefetch": str(coarse),
        "top_k": str(settings.phase6_visual_top_k),
    }

    if not coarse_hits:
        return [], telemetry

    top_k = settings.phase6_visual_top_k
    if not settings.phase6_visual_maxsim_rerank:
        telemetry["rerank"] = "dense_only"
        return _hits_cosine(coarse_hits, limit=top_k), telemetry

    hits, rerank_tag = _maxsim_rerank(
        query_vector,
        coarse_hits,
        slots=settings.phase6_maxsim_slots,
        top_k=top_k,
    )
    telemetry["rerank"] = rerank_tag
    return hits, telemetry
