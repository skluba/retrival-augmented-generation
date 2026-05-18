"""Repository-relative paths for bundled evaluation artifacts."""

from __future__ import annotations

from pathlib import Path

# …/src/rag_pdf_app/eval/paths.py → repo root is parents[3]
_REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_IFC_EVAL_CSV = _REPO_ROOT / "RAG_evaluation_dataset-convertcsv.csv"


def repo_root() -> Path:
    return _REPO_ROOT
