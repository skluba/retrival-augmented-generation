"""Render Phase 2 evaluation JSON payloads as GitHub-friendly Markdown."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isnan
from typing import Any

# Canonical column order when present (matches RAGAS metric names in reports).
_RAGAS_METRIC_KEYS = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)

_JUDGE_SCORE_KEYS = (
    "factual_alignment_with_reference",
    "completeness_vs_reference",
    "handling_of_visual_or_numeric_claims",
    "clarity",
    "overall",
)


def _md_cell(text: str, *, max_len: int = 500) -> str:
    """Escape minimal Markdown table breakage (pipes / newlines)."""

    s = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if len(s) > max_len:
        s = f"{s[: max_len - 12]}… [truncated]"
    return s.replace("|", "\\|").replace("\n", "<br>")


def _fmt_metric(v: object) -> str:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "—"
    if isnan(x):
        return "—"
    return f"{x:.3f}"


def _metric_keys_for_summary(mean: Mapping[str, Any]) -> list[str]:
    ordered = [k for k in _RAGAS_METRIC_KEYS if k in mean]
    extra = sorted(set(mean) - set(ordered))
    return ordered + extra


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body_lines = ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join([head, sep, *body_lines])


def _append_intro(
    lines: list[str],
    payload: Mapping[str, Any],
    *,
    source_json_basename: str | None,
) -> None:
    lines.append("# Phase 2 · RAG evaluation report")
    lines.append("")
    gen = str(payload.get("generated_at_utc", "")).strip()
    if gen:
        lines.append(f"**Generated (UTC):** `{_md_cell(gen, max_len=200)}`")
        lines.append("")
    if source_json_basename:
        lines.append(f"**Canonical JSON:** `{_md_cell(source_json_basename, max_len=256)}`")
        lines.append("")


def _append_ragas_summary(lines: list[str], mean: object) -> None:
    if not isinstance(mean, Mapping):
        return
    lines.append("## RAGAS · dataset mean")
    lines.append("")
    keys = _metric_keys_for_summary(mean)
    if keys:
        header = ["Metric", "Mean"]
        rows = [[_md_cell(k, max_len=120), _fmt_metric(mean[k])] for k in keys]
        lines.append(_markdown_table(header, rows))
    else:
        lines.append("_No summary metrics._")
    lines.append("")


def _collect_extra_columns(
    metric_cols: Sequence[str],
    records: Sequence[Mapping[str, Any]],
) -> list[str]:
    extra_cols: list[str] = []
    skip = {"context_content_type", *metric_cols}
    for rec in records:
        for key in rec:
            if key in skip or key in extra_cols:
                continue
            extra_cols.append(key)
    extra_cols.sort()
    return extra_cols


def _append_ragas_by_type(lines: list[str], raw: object) -> None:
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        return
    records = [r for r in raw if isinstance(r, Mapping)]
    lines.append("## RAGAS · mean by `Context_Content_Type`")
    lines.append("")
    if not records:
        lines.append("_No per-type breakdown._")
        lines.append("")
        return
    metric_cols = [k for k in _RAGAS_METRIC_KEYS if any(k in rec for rec in records)]
    extra_cols = _collect_extra_columns(metric_cols, records)
    headers = ["Context_Content_Type", *metric_cols, *extra_cols]
    rows_out: list[list[str]] = []
    for rec in records:
        ct = str(rec.get("context_content_type", "")).strip()
        row = [_md_cell(ct, max_len=200)]
        row.extend(_fmt_metric(rec.get(col)) for col in metric_cols)
        row.extend(_md_cell(str(rec.get(col, "")), max_len=80) for col in extra_cols)
        rows_out.append(row)
    lines.append(_markdown_table(headers, rows_out))
    lines.append("")


def _numeric_judge_row(scores: Mapping[str, Any]) -> dict[str, int] | None:
    row_scores: dict[str, int] = {}
    for k in _JUDGE_SCORE_KEYS:
        v = scores.get(k)
        if isinstance(v, bool):
            return None
        if isinstance(v, int | float):
            row_scores[k] = int(v)
        else:
            return None
    return row_scores


def _judge_failures(entries: Sequence[Any]) -> int:
    n = 0
    for e in entries:
        if not isinstance(e, Mapping):
            continue
        sc = e.get("scores")
        if isinstance(sc, Mapping) and "error" in sc:
            n += 1
    return n


def _append_judge(lines: list[str], raw: object) -> None:
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        return
    entries = list(raw)
    lines.append("## LLM judge")
    lines.append("")
    if not entries:
        lines.append("_Judge skipped or no rows matched heuristics._")
        lines.append("")
        return

    errors = _judge_failures(entries)
    lines.append(f"- Rows judged: **{len(entries)}**")
    lines.append(f"- Judge failures: **{errors}**")
    lines.append("")

    numeric_samples: list[dict[str, int]] = []
    for e in entries:
        if not isinstance(e, Mapping):
            continue
        sc = e.get("scores")
        if not isinstance(sc, Mapping) or "error" in sc:
            continue
        parsed = _numeric_judge_row(sc)
        if parsed is not None:
            numeric_samples.append(parsed)

    if numeric_samples:
        lines.append("Mean rubric scores (1–5, successful judge calls only):")
        lines.append("")
        header = ["Criterion", "Mean"]
        agg_rows: list[list[str]] = []
        for k in _JUDGE_SCORE_KEYS:
            vals = [s[k] for s in numeric_samples]
            mean_j = sum(vals) / len(vals)
            label = _md_cell(k.replace("_", " "), max_len=120)
            agg_rows.append([label, f"{mean_j:.2f}"])
        lines.append(_markdown_table(header, agg_rows))
    elif errors < len(entries):
        lines.append("_No numeric judge scores parsed._")
    lines.append("")


def _faithfulness_sort_key(row: Mapping[str, Any]) -> float | None:
    try:
        fv = float(row.get("faithfulness"))
    except (TypeError, ValueError):
        return None
    return None if isnan(fv) else fv


def _append_lowest_faithfulness(lines: list[str], raw: object, *, n: int) -> None:
    if n <= 0 or isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        return
    scored: list[tuple[float, Mapping[str, Any]]] = []
    for r in raw:
        if not isinstance(r, Mapping):
            continue
        fv = _faithfulness_sort_key(r)
        if fv is None:
            continue
        scored.append((fv, r))
    scored.sort(key=lambda t: t[0])
    worst = scored[:n]

    lines.append(f"## Lowest faithfulness (up to {n} rows)")
    lines.append("")
    if not worst:
        lines.append("_No per-row faithfulness scores available._")
        lines.append("")
        return

    headers = ["Faithfulness", "Context recall", "Page", "Type", "Question (truncated)"]
    rows_out: list[list[str]] = []
    for fv, r in worst:
        q = str(r.get("user_input", "")).strip()
        rows_out.append(
            [
                _fmt_metric(fv),
                _fmt_metric(r.get("context_recall")),
                _md_cell(str(r.get("page_number", "")).strip(), max_len=40),
                _md_cell(str(r.get("context_content_type", "")).strip(), max_len=120),
                _md_cell(q, max_len=160),
            ]
        )
    lines.append(_markdown_table(headers, rows_out))
    lines.append("")


def render_phase2_eval_markdown(
    payload: Mapping[str, Any],
    *,
    source_json_basename: str | None = None,
    lowest_faithfulness_n: int = 5,
) -> str:
    """Build Markdown from the evaluation payload dict (same schema as report JSON)."""

    lines: list[str] = []
    _append_intro(lines, payload, source_json_basename=source_json_basename)
    _append_ragas_summary(lines, payload.get("ragas_summary_mean"))
    _append_ragas_by_type(lines, payload.get("ragas_by_context_content_type"))
    _append_judge(lines, payload.get("llm_judge"))
    _append_lowest_faithfulness(lines, payload.get("per_row_ragas"), n=lowest_faithfulness_n)

    lines.append(
        "_Scores come from RAGAS + optional Gemini judge; interpret deltas against "
        "pipeline configuration and labeling assumptions._"
    )
    lines.append("")
    return "\n".join(lines)
