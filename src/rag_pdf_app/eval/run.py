"""CLI: Phase 2 RAG evaluation (RAGAS + optional LLM judge)."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import UTC, datetime
from pathlib import Path

from ragas.utils import safe_nanmean

from rag_pdf_app.config import clear_settings_cache, get_settings
from rag_pdf_app.eval.llm_judge import judge_answer_row, should_run_multimodal_judge
from rag_pdf_app.eval.load_csv import load_ifc_eval_csv
from rag_pdf_app.eval.markdown_report import render_phase2_eval_markdown
from rag_pdf_app.eval.paths import repo_root
from rag_pdf_app.eval.pipeline import (
    pipeline_rows_to_ragas_samples,
    run_phase1_on_eval_rows,
)
from rag_pdf_app.eval.ragas_runner import run_ragas_evaluation
from rag_pdf_app.eval.reporting import ragas_summary_by_content_type
from rag_pdf_app.rag.embeddings import vertex_text_embeddings
from rag_pdf_app.rag.stores import load_faiss_index


def _persist_phase2_reports(
    out_path: Path,
    payload: dict[str, object],
    *,
    write_markdown: bool,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote report: {out_path}", flush=True)
    if not write_markdown:
        return
    md_path = out_path.with_suffix(".md")
    md_path.write_text(
        render_phase2_eval_markdown(payload, source_json_basename=out_path.name),
        encoding="utf-8",
    )
    print(f"Wrote Markdown report: {md_path}", flush=True)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Phase 2 · Run IFC RAG evaluation: Phase 1 pipeline over labeled CSV, "
            "then RAGAS + optional Gemini judge."
        )
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Path to convertcsv-style evaluation file (default: repo root CSV).",
    )
    p.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Evaluate only the first N rows after loading (0 = all).",
    )
    p.add_argument(
        "--skip-judge",
        action="store_true",
        help="Skip LLM-as-judge scoring.",
    )
    p.add_argument(
        "--judge-all",
        action="store_true",
        help="Run judge on every row (default: only image/table/combination-like types).",
    )
    p.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Write full report JSON under reports/eval/ (default if omitted: auto path).",
    )
    p.add_argument(
        "--no-output-markdown",
        action="store_true",
        help="Do not write a sibling .md summary next to the JSON report.",
    )
    p.add_argument(
        "--ragas-timeout",
        type=int,
        default=240,
        help="Per-metric RAGAS timeout seconds (Vertex latency).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    clear_settings_cache()
    settings = get_settings()

    gold_rows = load_ifc_eval_csv(args.csv)
    if args.max_rows and args.max_rows > 0:
        gold_rows = gold_rows[: args.max_rows]

    if not gold_rows:
        print("No evaluation rows in CSV — nothing to run.", file=sys.stderr)
        return 2

    embedder = vertex_text_embeddings(settings)
    faiss_store = load_faiss_index(embedder, settings)

    print(f"Loaded {len(gold_rows)} evaluation rows; running Phase 1 pipeline…", flush=True)
    pipeline_rows = run_phase1_on_eval_rows(settings, faiss_store, gold_rows)
    ragas_samples = pipeline_rows_to_ragas_samples(pipeline_rows)

    print(
        "Running RAGAS (faithfulness, answer_relevancy, context_precision, context_recall)…",
        flush=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        ragas_result = run_ragas_evaluation(
            ragas_samples,
            settings,
            timeout_sec=args.ragas_timeout,
            raise_exceptions=False,
        )

    combined_df = ragas_result.to_pandas()
    metric_cols = [
        c
        for c in combined_df.columns
        if c
        in (
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
        )
    ]
    by_type = ragas_summary_by_content_type(combined_df, pipeline_rows, metric_cols)

    judge_rows: list[dict[str, object]] = []
    if not args.skip_judge:
        print("Running LLM judge…", flush=True)
        for pr in pipeline_rows:
            gold = pr.gold
            run_judge = args.judge_all or should_run_multimodal_judge(gold.context_content_type)
            if not run_judge:
                continue
            try:
                outcome = judge_answer_row(
                    settings,
                    question=gold.question,
                    reference_answer=gold.ground_truth_answer,
                    generated_answer=pr.response,
                    gold_context=gold.ground_truth_context,
                    content_type=gold.context_content_type,
                    page_number=gold.page_number,
                )
                scores = outcome.scores.model_dump()
            except Exception as exc:  # noqa: BLE001
                scores = {"error": str(exc)}
            judge_rows.append(
                {
                    "question": gold.question,
                    "context_content_type": gold.context_content_type,
                    "page_number": gold.page_number,
                    "scores": scores,
                }
            )

    out_path = args.output_json
    if out_path is None:
        reports = repo_root() / "reports" / "eval"
        reports.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
        out_path = reports / f"phase2_rag_eval_{stamp}.json"

    metric_keys = list(ragas_result.scores[0].keys()) if ragas_result.scores else []
    ragas_summary_mean = {
        k: float(safe_nanmean([row[k] for row in ragas_result.scores])) for k in metric_keys
    }

    payload: dict[str, object] = {
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "ragas_summary_mean": ragas_summary_mean,
        "ragas_by_context_content_type": by_type.reset_index().to_dict(orient="records"),
        "per_row_ragas": combined_df.assign(
            context_content_type=[r.gold.context_content_type for r in pipeline_rows],
            page_number=[r.gold.page_number for r in pipeline_rows],
        ).to_dict(orient="records"),
        "llm_judge": judge_rows,
    }

    _persist_phase2_reports(out_path, payload, write_markdown=not args.no_output_markdown)
    print(ragas_result, flush=True)
    print("\nMean RAGAS by Context_Content_Type:\n", by_type.to_string(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
