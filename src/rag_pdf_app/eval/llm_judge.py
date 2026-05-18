"""LLM-as-judge rubric for nuanced / multimodal-oriented QA."""

from __future__ import annotations

import json
from dataclasses import dataclass

from pydantic import BaseModel, Field, field_validator

from rag_pdf_app.config import Settings
from rag_pdf_app.vertex_gemini import generate_plain_text


class JudgeRubricScores(BaseModel):
    """1–5 integers returned by the judge model."""

    factual_alignment_with_reference: int = Field(
        ...,
        description="How well the generated answer matches key facts in the reference answer.",
    )
    completeness_vs_reference: int = Field(
        ...,
        description="Coverage of salient points compared to the reference (not verbatim match).",
    )
    handling_of_visual_or_numeric_claims: int = Field(
        ...,
        description=(
            "Appropriate treatment of chart/table-style facts when context type suggests them."
        ),
    )
    clarity: int = Field(..., description="Clear, direct prose.")
    overall: int = Field(..., description="Holistic usefulness for an analyst.")

    @field_validator(
        "factual_alignment_with_reference",
        "completeness_vs_reference",
        "handling_of_visual_or_numeric_claims",
        "clarity",
        "overall",
    )
    @classmethod
    def _v_range(cls, v: int) -> int:
        if not 1 <= v <= 5:
            raise ValueError("scores must be 1–5")
        return v


_JUDGE_PROMPT_TEMPLATE = """\
You evaluate answers from a retrieval-augmented system over IFC annual report material.
Score each criterion as an integer from 1 (poor) to 5 (excellent). Be strict about hallucinations.

Return ONLY a JSON object with exactly these keys and integer values:
"factual_alignment_with_reference",
"completeness_vs_reference",
"handling_of_visual_or_numeric_claims",
"clarity",
"overall"

Question:
{question}

Reference answer (gold):
{reference}

Generated answer:
{generated}

Declared gold context (may summarize visuals):
{gold_context}

Context content type label:
{content_type}

Pages (CSV metadata):
{pages}
"""


def _strip_markdown_fence(text: str) -> str:
    lines = text.strip().splitlines()
    if len(lines) >= 3 and lines[0].lstrip().startswith("```") and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return text.strip()


def _extract_json_object(text: str) -> dict[str, object]:
    body = _strip_markdown_fence(text)
    start = body.find("{")
    end = body.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in judge output")
    return json.loads(body[start : end + 1])


@dataclass
class JudgeOutcome:
    scores: JudgeRubricScores
    raw_text: str


def judge_answer_row(
    settings: Settings,
    *,
    question: str,
    reference_answer: str,
    generated_answer: str,
    gold_context: str,
    content_type: str,
    page_number: str,
) -> JudgeOutcome:
    """Gemini judge with a fixed rubric (good for image/table/combination questions)."""

    prompt = _JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        reference=reference_answer,
        generated=generated_answer,
        gold_context=gold_context[:8000],
        content_type=content_type,
        pages=page_number,
    )
    raw = generate_plain_text(prompt, settings, max_output_tokens=768)
    data = _extract_json_object(raw)
    scores = JudgeRubricScores.model_validate(data)
    return JudgeOutcome(scores=scores, raw_text=raw)


def should_run_multimodal_judge(content_type: str) -> bool:
    """Heuristic: run judge on rows likely tied to figures/tables/composites."""

    ct = content_type.lower()
    keywords = ("image", "figure", "visual", "table", "combination", "multi-modal", "multimodal")
    return any(k in ct for k in keywords)
