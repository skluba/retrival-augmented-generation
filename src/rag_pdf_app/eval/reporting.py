"""Aggregate evaluation outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from rag_pdf_app.config import Settings
from rag_pdf_app.eval.models import EvalPipelineRow


def _faiss_store_dir_basename(path_str: str) -> str:
    """Last path segment only (avoids leaking host layout in shared eval artifacts)."""

    p = Path(path_str).expanduser()
    return p.name if p.name else str(p)


def eval_retrieval_config_snapshot(settings: Settings) -> dict[str, Any]:
    """Stable dict embedded in eval JSON/Markdown so baseline vs hybrid runs are comparable."""

    faiss_path = settings.faiss_store_path
    snap: dict[str, Any] = {
        "rag_hybrid_enabled": settings.rag_hybrid_enabled,
        "rag_top_k": settings.rag_top_k,
        "rag_hybrid_dense_pool": settings.rag_hybrid_dense_pool,
        "rag_hybrid_sparse_pool": settings.rag_hybrid_sparse_pool,
        "rag_rrf_k": settings.rag_rrf_k,
        "rag_rrf_dense_weight": settings.rag_rrf_dense_weight,
        "rag_rrf_sparse_weight": settings.rag_rrf_sparse_weight,
        "rag_metadata_boost_enabled": settings.rag_metadata_boost_enabled,
        "rag_cross_encoder_model": settings.rag_cross_encoder_model,
        "rag_cross_encoder_top_n": settings.rag_cross_encoder_top_n,
        "rag_page_filter_min": settings.rag_page_filter_min,
        "rag_page_filter_max": settings.rag_page_filter_max,
        "faiss_store_basename": _faiss_store_dir_basename(faiss_path),
        "rag_qdrant_collection": settings.rag_qdrant_collection,
        "rag_semantic_cache_enabled": settings.rag_semantic_cache_enabled,
        "rag_semantic_cache_similarity_threshold": settings.rag_semantic_cache_similarity_threshold,
        "rag_multi_hop_enabled": settings.rag_multi_hop_enabled,
    }
    if settings.rag_eval_snapshot_include_absolute_paths:
        snap["faiss_store_path_absolute"] = str(Path(faiss_path).expanduser().resolve())
    return snap


def ragas_summary_by_content_type(
    df: pd.DataFrame,
    pipeline_rows: list[EvalPipelineRow],
    metric_cols: list[str],
) -> pd.DataFrame:
    """Mean RAGAS metric scores grouped by CSV ``Context_Content_Type``."""

    if len(df) != len(pipeline_rows):
        raise ValueError("DataFrame row count must match pipeline rows")

    ct = [r.gold.context_content_type for r in pipeline_rows]
    sub = df[metric_cols].copy()
    sub.insert(0, "context_content_type", ct)
    return sub.groupby("context_content_type", dropna=False)[metric_cols].mean()
