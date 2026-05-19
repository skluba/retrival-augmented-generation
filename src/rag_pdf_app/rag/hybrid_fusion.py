"""Reciprocal rank fusion and metadata filters for hybrid retrieval."""

from __future__ import annotations

from typing import Iterable

from rag_pdf_app.rag.models import RetrievalHit


def reciprocal_rank_fusion(rankings: list[list[str]], *, rrf_k: int = 60) -> dict[str, float]:
    """Standard RRF over ordered chunk-id lists (higher is better)."""

    scores: dict[str, float] = {}
    for ranked in rankings:
        for rank, cid in enumerate(ranked, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
    return scores


def env_page_window_to_zero_based(
    page_min_1based: int | None,
    page_max_1based: int | None,
) -> tuple[int, int] | None:
    """Convert optional 1-based inclusive PDF page bounds to 0-based inclusive metadata coords."""

    if page_min_1based is None and page_max_1based is None:
        return None
    lo = (page_min_1based - 1) if page_min_1based is not None else 0
    hi = (page_max_1based - 1) if page_max_1based is not None else 10**9
    return lo, hi


def intersect_page_windows(
    a: tuple[int, int] | None,
    b: tuple[int, int] | None,
) -> tuple[int, int] | None:
    if a is None:
        return b
    if b is None:
        return a
    lo = max(a[0], b[0])
    hi = min(a[1], b[1])
    if lo > hi:
        return None
    return lo, hi


def hit_overlaps_page_window(hit: RetrievalHit, window: tuple[int, int] | None) -> bool:
    if window is None:
        return True
    lo0, hi0 = window
    ps = int(hit.metadata.get("page_start", 0))
    pe = int(hit.metadata.get("page_end", ps))
    return pe >= lo0 and ps <= hi0


def filter_hits_by_page_window(
    hits: Iterable[RetrievalHit],
    window: tuple[int, int] | None,
) -> list[RetrievalHit]:
    if window is None:
        return list(hits)
    return [h for h in hits if hit_overlaps_page_window(h, window)]
