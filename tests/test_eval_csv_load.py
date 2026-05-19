"""Offline checks for Phase 2 evaluation CSV."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_pdf_app.eval.load_csv import load_ifc_eval_csv
from rag_pdf_app.eval.paths import DEFAULT_IFC_EVAL_CSV, repo_root

_FIXTURE_CSV = Path(__file__).resolve().parent / "fixtures" / "ifc_eval_sample.csv"


def test_load_fixture_csv() -> None:
    rows = load_ifc_eval_csv(_FIXTURE_CSV)
    assert len(rows) == 2
    assert "IFC" in rows[0].question


@pytest.mark.skipif(
    not DEFAULT_IFC_EVAL_CSV.is_file(),
    reason="Full IFC evaluation CSV not present at repo root",
)
def test_default_eval_csv_exists() -> None:
    assert DEFAULT_IFC_EVAL_CSV.is_file(), f"missing {DEFAULT_IFC_EVAL_CSV}"


@pytest.mark.skipif(
    not DEFAULT_IFC_EVAL_CSV.is_file(),
    reason="Full IFC evaluation CSV not present at repo root",
)
def test_load_ifc_eval_csv_row_shape() -> None:
    rows = load_ifc_eval_csv()
    assert len(rows) >= 20
    first = rows[0]
    assert "IFC" in first.question or "ifc" in first.question.lower()
    assert first.ground_truth_answer
    assert first.context_content_type


def test_repo_root_points_at_pyproject() -> None:
    root = repo_root()
    assert (root / "pyproject.toml").is_file()


@pytest.mark.parametrize(
    "filename",
    ["RAG_evaluation_dataset-convertcsv.csv"],
)
@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1] / "RAG_evaluation_dataset-convertcsv.csv").is_file(),
    reason="Full IFC evaluation CSV not present at repo root",
)
def test_csv_path_explicit(filename: str) -> None:
    path = Path(__file__).resolve().parents[1] / filename
    rows = load_ifc_eval_csv(path)
    assert rows
