"""Regression: duplicate PDF table headers must not break JSON export."""

import pandas as pd

from rag_pdf_app.parsing.tables import _dataframe_to_exports


def test_dataframe_to_exports_duplicate_columns() -> None:
    df = pd.DataFrame([[1, 2, 3]], columns=["Year", "Year", ""])
    _md, _csv, _html, js = _dataframe_to_exports(df.fillna(""))
    assert js == [{"Year": 1, "Year.1": 2, "": 3}]
