"""Aggregate evaluation outputs."""

from __future__ import annotations

import pandas as pd

from rag_pdf_app.eval.models import EvalPipelineRow


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
