"""Tests for Phase 2 evaluation Markdown rendering."""

from __future__ import annotations

from rag_pdf_app.eval.markdown_report import render_phase2_eval_markdown


def test_render_phase2_eval_markdown_includes_sections() -> None:
    payload = {
        "generated_at_utc": "2026-01-01T12:00:00+00:00",
        "retrieval_config": {
            "rag_hybrid_enabled": True,
            "rag_top_k": 5,
            "rag_rrf_sparse_weight": 1.15,
        },
        "ragas_summary_mean": {
            "faithfulness": 0.8,
            "answer_relevancy": 0.7,
            "context_precision": 0.75,
            "context_recall": 0.65,
        },
        "ragas_by_context_content_type": [
            {
                "context_content_type": "text",
                "faithfulness": 0.9,
                "answer_relevancy": 0.8,
                "context_precision": 0.85,
                "context_recall": 0.7,
            },
            {
                "context_content_type": "table",
                "faithfulness": 0.7,
                "answer_relevancy": 0.6,
                "context_precision": 0.65,
                "context_recall": 0.6,
            },
        ],
        "per_row_ragas": [
            {
                "user_input": "Q1?",
                "faithfulness": 0.5,
                "context_recall": 0.4,
                "page_number": "1",
                "context_content_type": "text",
            },
            {
                "user_input": "Q2?",
                "faithfulness": 0.9,
                "context_recall": 0.85,
                "page_number": "2",
                "context_content_type": "table",
            },
        ],
        "llm_judge": [
            {
                "question": "x",
                "context_content_type": "table",
                "page_number": "4",
                "scores": {
                    "factual_alignment_with_reference": 4,
                    "completeness_vs_reference": 3,
                    "handling_of_visual_or_numeric_claims": 4,
                    "clarity": 5,
                    "overall": 4,
                },
            },
            {
                "question": "y",
                "context_content_type": "image",
                "page_number": "6",
                "scores": {"error": "vertex timeout"},
            },
        ],
    }
    md = render_phase2_eval_markdown(payload, source_json_basename="phase2_rag_eval_smoke.json")
    assert "# Phase 2 · RAG evaluation report" in md
    assert "phase2_rag_eval_smoke.json" in md
    assert "## Retrieval configuration" in md
    assert "rag_hybrid_enabled" in md
    assert "## RAGAS · dataset mean" in md
    assert "## RAGAS · mean by `Context_Content_Type`" in md
    assert "| text |" in md
    assert "## LLM judge" in md
    assert "Rows judged: **2**" in md
    assert "Judge failures: **1**" in md
    assert "## Lowest faithfulness" in md
    assert "Q1?" in md


def test_render_phase2_eval_markdown_empty_judge() -> None:
    md = render_phase2_eval_markdown(
        {
            "generated_at_utc": "2026-01-01T00:00:00+00:00",
            "ragas_summary_mean": {},
            "ragas_by_context_content_type": [],
            "per_row_ragas": [],
            "llm_judge": [],
        }
    )
    assert "Judge skipped" in md or "no rows matched" in md
