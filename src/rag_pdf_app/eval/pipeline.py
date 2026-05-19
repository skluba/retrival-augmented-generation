"""Run Phase 1 RAG over evaluation questions and materialise rows for metrics."""

from __future__ import annotations

from langchain_community.vectorstores import FAISS

from rag_pdf_app.config import Settings
from rag_pdf_app.eval.models import EvalGoldRow, EvalPipelineRow
from rag_pdf_app.rag.query import run_phase1_rag
from rag_pdf_app.rag.retrieve import faiss_retrieved_chunk_texts


def run_phase1_on_eval_rows(
    settings: Settings,
    faiss_store: FAISS,
    rows: list[EvalGoldRow],
) -> list[EvalPipelineRow]:
    """Execute retrieval + generation for each gold question."""

    out: list[EvalPipelineRow] = []
    for row in rows:
        result = run_phase1_rag(settings, row.question, faiss_store=faiss_store)
        contexts = faiss_retrieved_chunk_texts(result.retrieval)
        out.append(
            EvalPipelineRow(
                gold=row,
                response=result.answer,
                retrieved_contexts=contexts,
                ragas_extra={
                    "faiss_latency_ms": result.retrieval.faiss_timing.latency_ms,
                    "qdrant_latency_ms": result.retrieval.qdrant_timing.latency_ms,
                    "dual_retrieval_notes": list(result.retrieval.notes),
                    "rag_hybrid_enabled": settings.rag_hybrid_enabled,
                },
            )
        )
    return out


def pipeline_rows_to_ragas_samples(rows: list[EvalPipelineRow]) -> list[dict[str, object]]:
    """Build dicts for ``ragas.EvaluationDataset.from_list`` (single-turn)."""

    samples: list[dict[str, object]] = []
    for r in rows:
        samples.append(
            {
                "user_input": r.gold.question,
                "retrieved_contexts": r.retrieved_contexts,
                "response": r.response,
                "reference": r.gold.ground_truth_answer,
                "reference_contexts": [r.gold.ground_truth_context],
            }
        )
    return samples
