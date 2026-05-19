"""RAGAS metrics over Phase 1 outputs (legacy metric bundle compatible with Vertex LLM)."""

from __future__ import annotations

import warnings

from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics._answer_relevance import answer_relevancy
from ragas.metrics._context_precision import context_precision
from ragas.metrics._context_recall import context_recall
from ragas.metrics._faithfulness import faithfulness
from ragas.run_config import RunConfig

from rag_pdf_app.config import Settings


def vertex_ragas_llm(settings: Settings, *, temperature: float = 0.2) -> LangchainLLMWrapper:
    """LangChain Vertex chat model wrapped for RAGAS metric prompts."""

    lc = ChatVertexAI(
        model=settings.vertex_generative_model,
        project=settings.google_cloud_project,
        location=settings.google_cloud_location,
        temperature=temperature,
    )
    return LangchainLLMWrapper(lc)


def vertex_ragas_embeddings(settings: Settings) -> VertexAIEmbeddings:
    return VertexAIEmbeddings(
        project=settings.google_cloud_project,
        location=settings.google_cloud_location,
        model_name=settings.vertex_embedding_model,
    )


def default_ragas_metrics():
    """Faithfulness, answer relevancy, context precision (vs reference), context recall."""

    return [faithfulness, answer_relevancy, context_precision, context_recall]


def run_ragas_evaluation(
    samples: list[dict[str, object]],
    settings: Settings,
    *,
    timeout_sec: int = 240,
    raise_exceptions: bool = False,
):
    """Evaluate prepared single-turn samples; returns RAGAS ``EvaluationResult``."""

    ds = EvaluationDataset.from_list(samples)
    llm = vertex_ragas_llm(settings)
    embeddings = vertex_ragas_embeddings(settings)
    rc = RunConfig(timeout=timeout_sec)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return evaluate(
            ds,
            metrics=default_ragas_metrics(),
            llm=llm,
            embeddings=embeddings,
            run_config=rc,
            raise_exceptions=raise_exceptions,
        )
