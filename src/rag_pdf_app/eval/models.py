"""Typed rows for Phase 2 IFC RAG evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EvalGoldRow:
    """One labeled Q&A row from the IFC evaluation CSV."""

    question: str
    ground_truth_context: str
    ground_truth_answer: str
    page_number: str
    context_content_type: str


@dataclass
class EvalPipelineRow:
    """Gold row plus Phase 1 outputs (for RAGAS + reporting)."""

    gold: EvalGoldRow
    response: str
    retrieved_contexts: list[str]
    ragas_extra: dict[str, object] = field(default_factory=dict)
