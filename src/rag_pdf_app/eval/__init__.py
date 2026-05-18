"""Phase 2 · RAG evaluation (RAGAS + LLM judge)."""

from rag_pdf_app.eval.load_csv import load_ifc_eval_csv
from rag_pdf_app.eval.pipeline import (
    pipeline_rows_to_ragas_samples,
    run_phase1_on_eval_rows,
)
from rag_pdf_app.eval.run import main as run_evaluation_cli

__all__ = [
    "load_ifc_eval_csv",
    "pipeline_rows_to_ragas_samples",
    "run_evaluation_cli",
    "run_phase1_on_eval_rows",
]
