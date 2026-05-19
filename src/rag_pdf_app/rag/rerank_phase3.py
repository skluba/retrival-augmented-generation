"""Phase 3 re-ranking: metadata-aware boosts + optional cross-encoder scores."""

from __future__ import annotations

import logging

from rag_pdf_app.rag.sparse_bm25 import tokenize
from rag_pdf_app.rag.models import RetrievalHit

_LOG = logging.getLogger(__name__)

_CE_MODELS: dict[str, object] = {}


def metadata_boost_rerank(hits: list[RetrievalHit], query: str) -> list[RetrievalHit]:
    """Lightweight lexicon overlap between query and ``content_type`` / cue keywords."""

    q_low = query.lower()
    q_tokens = set(tokenize(query))

    def bonus(hit: RetrievalHit) -> float:
        b = 0.0
        ct = str(hit.metadata.get("content_type", "")).lower()
        if "figure" in q_low and "figure" in ct:
            b += 1.5
        if "table" in q_low and "table" in ct:
            b += 1.5
        sh = str(hit.metadata.get("section_hint") or "").lower()
        if sh:
            overlap = sum(1 for t in q_tokens if len(t) > 3 and t in sh)
            b += 0.25 * overlap
        return b

    decorated = [(bonus(h), i, h) for i, h in enumerate(hits)]
    decorated.sort(key=lambda t: (-t[0], t[1]))
    return [h for _, __, h in decorated]


def cross_encoder_rerank(
    query: str,
    hits: list[RetrievalHit],
    *,
    model_name: str,
) -> list[RetrievalHit]:
    """Re-score (query, passage) pairs with a cross-encoder (requires optional ``phase3`` extra)."""

    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Cross-encoder reranking requires the optional dependency group "
            "``phase3`` (``sentence-transformers``). Install with "
            "``uv sync --extra phase3`` or ``pip install sentence-transformers``."
        ) from exc

    if model_name not in _CE_MODELS:
        _LOG.info("loading cross-encoder %s", model_name)
        _CE_MODELS[model_name] = CrossEncoder(model_name)
    model = _CE_MODELS[model_name]
    pairs = [(query, h.text) for h in hits]
    raw_scores = model.predict(pairs)
    scored = sorted(
        zip(raw_scores, hits, strict=True),
        key=lambda t: float(t[0]),
        reverse=True,
    )
    return [h for _, h in scored]
