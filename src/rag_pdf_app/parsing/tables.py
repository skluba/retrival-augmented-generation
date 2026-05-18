"""Table extraction via Camelot (preferred) plus PyMuPDF ``find_tables`` fallback."""

from __future__ import annotations

import hashlib
import io
import json
import uuid
import warnings
from collections.abc import Iterable, Sequence
from contextlib import redirect_stderr
from typing import Any, cast

import fitz  # PyMuPDF
import pandas as pd

from rag_pdf_app.parsing.models import BBox, TableBlock

type ExportBundle = tuple[str | None, str | None, str | None, list[dict[str, Any]] | None]


def _dataframe_to_exports(df: pd.DataFrame) -> ExportBundle:
    if df.empty:
        return None, None, None, None
    csv_s = df.to_csv(index=False)
    html_s = df.to_html(index=False)
    md_s: str | None = None
    try:
        md_s = df.to_markdown(index=False)
    except ImportError:
        md_s = None
    except ValueError:
        md_s = None
    if md_s is None:
        md_s = f"```csv\n{csv_s}\n```"
    try:
        as_json_list = json.loads(df.to_json(orient="records", default_handler=str))
    except (json.JSONDecodeError, TypeError):
        as_json_list = None
    return md_s, csv_s, html_s or None, cast(list[dict[str, Any]] | None, as_json_list)


def _rows_heuristic_df(rows_raw: Iterable[Sequence[Any]]) -> pd.DataFrame:
    rows_all = [[("" if v is None else v) for v in row] for row in rows_raw]
    if not rows_all:
        return pd.DataFrame()
    header = rows_all[0]
    cols = [str(h) if h not in {"", None} else "" for h in header]
    if len(rows_all) == 1:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows_all[1:], columns=cols)


CellValue = str | float | None


def _dataframe_to_cells(df: pd.DataFrame) -> list[list[CellValue]]:
    out: list[list[CellValue]] = []
    df2 = df.replace({pd.NaT: None})
    for _, row in df2.iterrows():
        r: list[CellValue] = []
        for raw in row.tolist():
            if pd.isna(raw):
                r.append(None)
            else:
                r.append(cast(CellValue, str(raw)))
        out.append(r)
    return out


def _camelot_read_all_pages(pdf_path: str, flavor: str) -> Any:
    import camelot  # type: ignore[import-untyped]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with redirect_stderr(io.StringIO()):
            return camelot.read_pdf(pdf_path, pages="all", flavor=flavor)  # type: ignore[operator]


def _table_block_from_camelot_table(
    t: Any,
    *,
    extractor_label: str,
    flavor: str,
    tbl_idx: int,
    seen_cells: set[tuple[str, int]],
) -> TableBlock | None:
    df = getattr(t, "df", None)
    if df is None or df.empty:
        return None
    hasher = hashlib.sha256()
    for _, row in df.iterrows():
        txt = "|".join(row.astype(str).tolist()).encode("utf-8", errors="replace")
        hasher.update(txt)
        hasher.update(b"\n")
    digest = hasher.hexdigest()
    page_one_based = getattr(t, "page", None)
    page_idx = max(int(page_one_based) - 1, 0) if page_one_based is not None else 0
    dedupe_key = (digest, page_idx)
    if dedupe_key in seen_cells:
        return None
    seen_cells.add(dedupe_key)

    md_s, csv_s, html_s, js = _dataframe_to_exports(df.fillna(""))
    rows_cells = _dataframe_to_cells(df)

    bbox_vals = getattr(t, "_bbox", None)
    bbox_obj: BBox | None = None
    camelot_box_tuple: tuple[float, float, float, float] | None = None
    if bbox_vals is not None and isinstance(bbox_vals, (tuple, list)):
        coords = bbox_vals if isinstance(bbox_vals, list) else list(bbox_vals)  # type: ignore[list-item,misc]
        if len(coords) == 4:
            x1, y1, x2, y2 = (float(v) for v in coords)
            camelot_box_tuple = (x1, y1, x2, y2)
            bbox_obj = BBox(
                page_index=page_idx,
                x0=min(x1, x2),
                y0=min(y1, y2),
                x1=max(x1, x2),
                y1=max(y1, y2),
            )

    parsing_report = getattr(t, "parsing_report", {})
    report_val = parsing_report if isinstance(parsing_report, dict) else str(parsing_report)

    return TableBlock(
        table_id=f"camelot-{extractor_label}-p{page_idx}-{tbl_idx}-{uuid.uuid4().hex[:6]}",
        page_index=page_idx,
        bbox=bbox_obj,
        extractor=extractor_label,  # type: ignore[arg-type]
        rows=rows_cells,
        as_markdown=md_s,
        as_csv=csv_s,
        as_html=html_s,
        as_json=js,
        metadata={
            "camelot_flavor": flavor,
            "camelot_bbox_raw": camelot_box_tuple,
            "camelot_accuracy": getattr(t, "accuracy", None),
            "camelot_whitespace": getattr(t, "whitespace", None),
            "camelot_report": report_val,
        },
    )


def extract_tables_camelot(pdf_path: str, notes: list[str]) -> list[TableBlock]:
    try:
        import camelot  # noqa: F401
    except ImportError as exc:
        notes.append(f"camelot_unavailable:{exc}")
        return []

    out: list[TableBlock] = []
    seen_cells: set[tuple[str, int]] = set()

    for flavor, extractor_label in (("lattice", "camelot_lattice"), ("stream", "camelot_stream")):
        try:
            cand = _camelot_read_all_pages(pdf_path, flavor)
        except Exception as exc:  # noqa: BLE001
            notes.append(f"camelot_{flavor}_failed:{exc}")
            continue
        for tbl_idx in range(len(cand)):
            t = cand[tbl_idx]
            block = _table_block_from_camelot_table(
                t,
                extractor_label=extractor_label,
                flavor=flavor,
                tbl_idx=tbl_idx,
                seen_cells=seen_cells,
            )
            if block is not None:
                out.append(block)
    return out


def _pymupdf_table_to_block(
    tbl: Any,
    *,
    pi: int,
    ti: int,
    notes: list[str],
) -> TableBlock | None:
    try:
        rows_raw = tbl.extract()  # type: ignore[attr-defined,no-untyped-call]
    except (RuntimeError, ValueError, AttributeError) as exc:
        notes.append(f"pymupdf_table_extract_failed_p{pi}_t{ti}:{exc}")
        return None
    if not rows_raw:
        return None
    df = _rows_heuristic_df(rows_raw)
    md_s, csv_s, html_s, js = _dataframe_to_exports(df.fillna(""))
    rows_cells = _dataframe_to_cells(df)

    bbox_obj: BBox | None = None
    rect = getattr(tbl, "bbox", None)
    if rect is not None and len(rect) == 4:  # type: ignore[arg-type]
        xs0, ys0, xs1, ys1 = (float(v) for v in rect)  # type: ignore[misc]
        bbox_obj = BBox(page_index=pi, x0=xs0, y0=ys0, x1=xs1, y1=ys1)

    return TableBlock(
        table_id=f"pymupdf-ft-p{pi}-t{ti}-{uuid.uuid4().hex[:6]}",
        page_index=pi,
        bbox=bbox_obj,
        extractor="pymupdf_find_tables",
        rows=rows_cells,
        as_markdown=md_s,
        as_csv=csv_s,
        as_html=html_s,
        as_json=js,
        metadata={"pymupdf_table_index": ti},
    )


def _tables_from_pymupdf_page(doc: fitz.Document, pi: int, notes: list[str]) -> list[TableBlock]:
    page = doc[pi]
    finder_builder = getattr(page, "find_tables", None)
    if not callable(finder_builder):
        return []
    try:
        table_finder = finder_builder()
    except Exception as exc:  # noqa: BLE001
        notes.append(f"pymupdf_find_tables_page_{pi}:{exc}")
        return []

    pym_tables = getattr(table_finder, "tables", None)
    if not pym_tables:
        return []

    out: list[TableBlock] = []
    for ti, tbl in enumerate(pym_tables):
        block = _pymupdf_table_to_block(tbl, pi=pi, ti=ti, notes=notes)
        if block is not None:
            out.append(block)
    return out


def extract_tables_pymupdf(pdf_bytes: bytes, notes: list[str]) -> list[TableBlock]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    out: list[TableBlock] = []

    try:
        for pi in range(len(doc)):
            out.extend(_tables_from_pymupdf_page(doc, pi, notes))
    finally:
        doc.close()
    return out
