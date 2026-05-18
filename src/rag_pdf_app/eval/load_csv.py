"""Load the bundled IFC RAG evaluation CSV."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from rag_pdf_app.eval.models import EvalGoldRow
from rag_pdf_app.eval.paths import DEFAULT_IFC_EVAL_CSV


def load_ifc_eval_csv(path: Path | None = None) -> list[EvalGoldRow]:
    """Parse evaluation CSV into structured rows (skips blank lines)."""

    csv_path = Path(path) if path is not None else DEFAULT_IFC_EVAL_CSV
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Evaluation CSV not found at {csv_path}. "
            "Place `RAG_evaluation_dataset - convertcsv.csv` at the repository root "
            "or pass --csv."
        )

    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    expected = {
        "Question",
        "Ground_Truth_Context",
        "Ground_Truth_Answer",
        "Page_Number",
        "Context_Content_Type",
    }
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns {sorted(missing)}; got {list(df.columns)}")

    rows: list[EvalGoldRow] = []
    for _, r in df.iterrows():
        q = str(r["Question"]).strip()
        if not q:
            continue
        rows.append(
            EvalGoldRow(
                question=q,
                ground_truth_context=str(r["Ground_Truth_Context"]).strip(),
                ground_truth_answer=str(r["Ground_Truth_Answer"]).strip(),
                page_number=str(r["Page_Number"]).strip(),
                context_content_type=str(r["Context_Content_Type"]).strip(),
            )
        )
    return rows
