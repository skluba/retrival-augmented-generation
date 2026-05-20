"""Optional second retrieval hop guided by the LLM (Phase 4).

Flow: run baseline dual retrieval → ask Gemini for either ``DONE`` or a short follow-up search
query → retrieve again → merge FAISS hits by ``chunk_id`` (Qdrant leg stays the first pass for a
stable baseline column in the UI).
"""

from __future__ import annotations

import hashlib
import logging

from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from qdrant_client import QdrantClient

from rag_pdf_app.config import Settings
from rag_pdf_app.rag.models import RetrievalHit
from rag_pdf_app.rag.retrieve import BackendTiming, DualRetrievalResult, retrieve_dual
from rag_pdf_app.vertex_gemini import generate_plain_text

_LOG = logging.getLogger(__name__)


def _refinement_prompt(user_question: str, context_snippet: str) -> str:
    snippet = context_snippet.strip()
    if len(snippet) > 3500:
        snippet = f"{snippet[:3497]}..."
    return (
        "You assist retrieval for a financial-report RAG system.\n"
        "Given the user's question and short excerpts from a first search, reply with "
        "**exactly one line**:\n"
        "- If these excerpts are enough to answer faithfully, output: DONE\n"
        "- Otherwise output **one** concise search query (no quotes, no bullets) to "
        "retrieve missing facts.\n\n"
        f"Question:\n{user_question.strip()}\n\n"
        "First-pass excerpts (untrusted document text):\n"
        f"{snippet}\n\n"
        "One line:"
    )


def merge_faiss_hits_deduped(
    primary: list[RetrievalHit],
    secondary: list[RetrievalHit],
    *,
    top_k: int,
) -> list[RetrievalHit]:
    seen: set[str] = set()
    out: list[RetrievalHit] = []
    for h in (*primary, *secondary):
        cid = (h.chunk_id or "").strip()
        dedupe_key = cid if cid else str(id(h))
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        out.append(h)
        if len(out) >= top_k:
            break
    return out


def retrieve_dual_with_multi_hop(
    *,
    settings: Settings,
    embeddings: Embeddings,
    faiss_store: FAISS,
    qdrant: QdrantClient,
    retrieval_query: str,
    collection: str,
    top_k: int,
    inline_page_window_1based: tuple[int, int] | None,
    original_question: str,
) -> tuple[DualRetrievalResult, bool]:
    """First retrieval pass; optionally refine query and merge FAISS lists."""

    dual1 = retrieve_dual(
        query=retrieval_query,
        embeddings=embeddings,
        faiss_store=faiss_store,
        qdrant=qdrant,
        collection=collection,
        top_k=top_k,
        settings=settings,
        inline_page_window_1based=inline_page_window_1based,
    )

    if not settings.rag_multi_hop_enabled:
        return dual1, False

    ctx_snippet = "\n\n".join(h.text[:800] for h in dual1.faiss_hits[:4])
    if not ctx_snippet.strip():
        notes = [*dual1.notes, "multi_hop_skipped:empty_first_pass"]
        return (
            DualRetrievalResult(
                faiss_hits=dual1.faiss_hits,
                qdrant_hits=dual1.qdrant_hits,
                faiss_timing=dual1.faiss_timing,
                qdrant_timing=dual1.qdrant_timing,
                notes=notes,
            ),
            False,
        )

    try:
        suggestion = generate_plain_text(
            _refinement_prompt(original_question, ctx_snippet),
            settings,
            max_output_tokens=128,
        )
    except Exception:
        _LOG.warning("multi-hop refinement LLM call failed", exc_info=True)
        notes = [*dual1.notes, "multi_hop_skipped:refinement_error"]
        return (
            DualRetrievalResult(
                faiss_hits=dual1.faiss_hits,
                qdrant_hits=dual1.qdrant_hits,
                faiss_timing=dual1.faiss_timing,
                qdrant_timing=dual1.qdrant_timing,
                notes=notes,
            ),
            False,
        )

    line = suggestion.splitlines()[0].strip() if suggestion else ""
    if not line or line.upper() == "DONE":
        notes = [*dual1.notes, "multi_hop_skipped:done"]
        return (
            DualRetrievalResult(
                faiss_hits=dual1.faiss_hits,
                qdrant_hits=dual1.qdrant_hits,
                faiss_timing=dual1.faiss_timing,
                qdrant_timing=dual1.qdrant_timing,
                notes=notes,
            ),
            False,
        )

    follow_q = line[:512]
    _LOG.debug("multi-hop follow-up query (not exposed in UI notes): %s", follow_q)
    dual2 = retrieve_dual(
        query=follow_q,
        embeddings=embeddings,
        faiss_store=faiss_store,
        qdrant=qdrant,
        collection=collection,
        top_k=top_k,
        settings=settings,
        inline_page_window_1based=inline_page_window_1based,
    )

    merged = merge_faiss_hits_deduped(
        dual1.faiss_hits,
        dual2.faiss_hits,
        top_k=top_k,
    )
    fq_digest = hashlib.sha256(follow_q.encode("utf-8", errors="replace")).hexdigest()
    notes = [
        *dual1.notes,
        "multi_hop_second_pass",
        f"multi_hop_follow_query_sha256:{fq_digest}",
        f"multi_hop_follow_query_len:{len(follow_q)}",
        f"multi_hop_merged_faiss_count:{len(merged)}",
    ]
    return (
        DualRetrievalResult(
            faiss_hits=merged,
            qdrant_hits=dual1.qdrant_hits,
            faiss_timing=BackendTiming(
                latency_ms=dual1.faiss_timing.latency_ms + dual2.faiss_timing.latency_ms,
                metric=dual1.faiss_timing.metric,
            ),
            qdrant_timing=dual1.qdrant_timing,
            notes=notes,
        ),
        True,
    )
