"""Turn parsed PDF tables into embeddable text chunks (Phase 5.1)."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from typing import Literal

from rag_pdf_app.parsing.models import ParsedPdf, TableBlock
from rag_pdf_app.rag.models import TextChunk


def _chunk_id_for_table(tbl: TableBlock, pdf_digest: str) -> str:
    blob = f"{pdf_digest}\x00{tbl.table_id}\x00{tbl.page_index}".encode()
    return hashlib.sha256(blob).hexdigest()[:24]


def _rows_to_markdown(rows: list[list[str | float | None]], *, max_rows: int = 80) -> str:
    lines: list[str] = []
    for i, row in enumerate(rows[:max_rows]):
        cells = ["" if c is None else str(c).strip() for c in row]
        lines.append("| " + " | ".join(cells) + " |")
        if i == 0:
            lines.append("| " + " | ".join("---" for _ in cells) + " |")
    return "\n".join(lines)


def _table_body(
    tbl: TableBlock,
    *,
    max_chars: int,
) -> str:
    md = (tbl.as_markdown or "").strip()
    if md:
        return md[:max_chars]
    csv_text = (tbl.as_csv or "").strip()
    if csv_text:
        return csv_text[:max_chars]
    if tbl.rows:
        return _rows_to_markdown(tbl.rows)[:max_chars]
    return ""


def table_text_chunk_from_block(
    tbl: TableBlock,
    *,
    pdf_sha256: str,
    source: str,
    max_text_chars: int = 12_000,
    csv_preview_chars: int = 6_000,
) -> TextChunk | None:
    """Build one searchable chunk for a :class:`TableBlock`, or ``None`` if empty."""

    body = _table_body(tbl, max_chars=max_text_chars)
    head_bits: list[str] = [
        f"TABLE · page {tbl.page_index + 1} · {tbl.extractor}",
        f"table_id={tbl.table_id}",
    ]
    summary = (tbl.summary or "").strip()
    if summary:
        head_bits.append(f"Summary: {summary}")

    near = tbl.metadata.get("contextual_snippet_above") or ""
    if isinstance(near, str) and near.strip():
        head_bits.append(f"Nearby text (above): {near.strip()[:400]}")

    header = "\n".join(head_bits)
    full_text = f"{header}\n\n{body}".strip()
    if not full_text or not body:
        return None

    cid = _chunk_id_for_table(tbl, pdf_sha256)
    csv_preview: str | None = None
    if tbl.as_csv and tbl.as_csv.strip():
        csv_preview = tbl.as_csv.strip()[:csv_preview_chars]

    return TextChunk(
        chunk_id=cid,
        text=full_text[:max_text_chars],
        page_start=tbl.page_index,
        page_end=tbl.page_index,
        source=source,
        structure_note=f"pdf_table:{tbl.extractor}",
        content_type="table_structured",
        section_hint=summary.split("\n", 1)[0][:240] if summary else f"page {tbl.page_index + 1}",
        chunk_kind="pdf_table",
        table_id=tbl.table_id,
        table_csv_preview=csv_preview,
    )


def table_text_chunks_from_parsed_pdf(
    parsed: ParsedPdf,
    *,
    max_text_chars: int = 12_000,
    csv_preview_chars: int = 6_000,
) -> list[TextChunk]:
    """Materialise one :class:`TextChunk` per extracted table."""

    out: list[TextChunk] = []
    for tbl in parsed.tables:
        ch = table_text_chunk_from_block(
            tbl,
            pdf_sha256=parsed.pdf_bytes_sha256,
            source=parsed.filename,
            max_text_chars=max_text_chars,
            csv_preview_chars=csv_preview_chars,
        )
        if ch is not None:
            out.append(ch)
    return out


def merge_narrative_and_table_chunks(
    narrative: Iterable[TextChunk],
    tables: Iterable[TextChunk],
) -> tuple[list[TextChunk], Literal["narrative_only", "merged"]]:
    """Append table chunks after narrative windows (shared index / BM25 corpus)."""

    narr = list(narrative)
    tblocks = list(tables)
    if not tblocks:
        return narr, "narrative_only"
    return [*narr, *tblocks], "merged"
