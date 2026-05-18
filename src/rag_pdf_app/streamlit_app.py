"""Streamlit entrypoint for environment checks and PDF ingestion lab."""

from __future__ import annotations

import base64
import json
import os
from typing import Any

import streamlit as st

from rag_pdf_app.config import Settings, clear_settings_cache, get_settings

st.set_page_config(page_title="PDF RAG Lab", layout="wide")
st.title("PDF RAG workspace")
st.caption("Gemini · Vertex AI · PyMuPDF · pdfminer · pypdf · Docling · Camelot")


def _scrub_b64_for_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        cleaned: dict[str, Any] = {}
        for k, v in obj.items():
            if k == "image_bytes_b64" and isinstance(v, str) and len(v) > 64:
                cleaned[k] = f"<<base64 omitted, {len(v)} chars>>"
            else:
                cleaned[k] = _scrub_b64_for_json(v)
        return cleaned
    if isinstance(obj, list):
        return [_scrub_b64_for_json(item) for item in obj]
    return obj


with st.sidebar:
    st.markdown("### Environment")
    st.code(
        (
            "Use GCP Application Default Credentials — no Gemini API keys.\n"
            "Example: `gcloud auth application-default login`"
        ),
        language="text",
    )
    if st.button("Reload `.env` / env"):
        clear_settings_cache()
        st.rerun()

settings_obj: Settings | None = None

try:
    settings_obj = get_settings()
except Exception as exc:  # noqa: BLE001
    st.error("Missing or invalid environment configuration.")
    st.exception(exc)
else:
    overview_tab, parsing_tab = st.tabs(["Overview", "PDF parsing"])

    with overview_tab:
        st.success("Loaded settings from environment.")
        st.json(
            {
                "GOOGLE_CLOUD_PROJECT": settings_obj.google_cloud_project,
                "GOOGLE_CLOUD_LOCATION": settings_obj.google_cloud_location,
                "GOOGLE_GENAI_USE_VERTEXAI": settings_obj.use_vertex_ai,
                "VERTEX_GENERATIVE_MODEL": settings_obj.vertex_generative_model,
                "QDRANT_URL": settings_obj.qdrant_url,
                "LANGFUSE_HOST": settings_obj.langfuse_host,
                "LANGFUSE_PUBLIC_KEY_set": settings_obj.langfuse_public_key is not None,
                "LANGFUSE_SECRET_KEY_set": settings_obj.langfuse_secret_key is not None,
                "GOOGLE_APPLICATION_CREDENTIALS_set": bool(
                    os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
                ),
            }
        )

        if st.checkbox("Smoke test Gemini (calls Vertex AI)"):
            from rag_pdf_app.vertex_gemini import generate_plain_text

            prompt = st.text_area("Prompt", value="Respond with exactly: pong")
            if st.button("Run Gemini text"):
                with st.spinner("Calling Gemini…"):
                    out = generate_plain_text(prompt, settings_obj)
                st.write(out)

        st.markdown("Next: chunk + embed + Langfuse tracing + RAGAS once retrieval wiring lands.")

    with parsing_tab:
        from rag_pdf_app.parsing.pipeline import parse_pdf_bytes

        st.subheader("Structured PDF parsing")

        run_docling = st.checkbox("Run Docling (slower, richer structure)", value=False)
        run_camelot = st.checkbox(
            "Camelot lattice/stream tables (Ghostscript — trusted PDFs only)",
            value=False,
            help=(
                "Camelot shells out to Ghostscript on a tempfile. Disable for untrusted uploads "
                "because native parsers have historically had memory-safety issues."
            ),
        )
        gemini_caps = st.checkbox(
            "Gemini captions for raster images (multimodal quota)", value=True
        )
        gemini_tbl = st.checkbox("Gemini summaries for detected tables", value=True)
        embed_b64 = st.checkbox("Embed base64 payloads in output JSON", value=True)

        pdf_file = st.file_uploader(
            "Upload a PDF",
            type=["pdf"],
            help="Text is extracted with pypdf, pdfminer layout, and optionally Docling.",
        )

        parse_clicked = st.button(
            "Parse PDF",
            type="primary",
            disabled=(pdf_file is None),
            help="Choose a PDF file first." if pdf_file is None else None,
        )
        if pdf_file is not None and parse_clicked:
            raw = pdf_file.getvalue()
            with st.spinner("Running extractors (this can take a while)…"):
                parsed = parse_pdf_bytes(
                    raw,
                    pdf_file.name,
                    settings=settings_obj,
                    embed_image_base64=embed_b64,
                    gemini_image_captions=gemini_caps,
                    gemini_table_summaries=gemini_tbl,
                    run_docling=run_docling,
                    run_camelot=run_camelot,
                )
            st.session_state["last_parsed_pdf"] = parsed

        parsed_state = st.session_state.get("last_parsed_pdf")
        if parsed_state is not None:
            st.markdown("### Artefact summary")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Text spans", len(parsed_state.text_spans))
            with c2:
                st.metric("Images", len(parsed_state.images))
            with c3:
                st.metric("Tables", len(parsed_state.tables))
            with c4:
                st.metric("Parse notes", len(parsed_state.parsing_notes))

            if parsed_state.parsing_notes:
                with st.expander("Parser warnings"):
                    for line in parsed_state.parsing_notes:
                        st.markdown(f"- `{line}`")

            with st.expander("PDF file metadata (pypdf)"):
                st.json(parsed_state.file_level_metadata)

            st.markdown("#### Figure previews")
            for fig in parsed_state.images[:24]:
                cols = st.columns([1, 2])
                with cols[0]:
                    blob = base64.b64decode(fig.image_bytes_b64) if fig.image_bytes_b64 else None
                    if blob:
                        st.image(blob, caption=f"xref {fig.xref} · page {fig.page_index + 1}")
                    else:
                        st.caption("No raster bytes inlined (toggle base64)")
                with cols[1]:
                    st.write(fig.metadata)
                    if fig.related_text_above_span_id or fig.related_text_below_span_id:
                        st.caption(
                            f"Links · above `{fig.related_text_above_span_id}` · below "
                            f"`{fig.related_text_below_span_id}`"
                        )
                    if fig.caption:
                        st.info(fig.caption)

            st.markdown("#### Tables")
            for tbl in parsed_state.tables[:12]:
                st.markdown(f"**{tbl.table_id}** — `{tbl.extractor}` — page {tbl.page_index + 1}")
                if tbl.as_markdown:
                    st.markdown(tbl.as_markdown)
                elif tbl.as_csv:
                    st.code(tbl.as_csv, language="csv")
                if tbl.summary:
                    st.success(tbl.summary)
                if tbl.metadata.get("contextual_snippet_above") or tbl.metadata.get(
                    "contextual_snippet_below"
                ):
                    st.caption(
                        "Context · above: "
                        f"{tbl.metadata.get('contextual_snippet_above')}"
                        " · below: "
                        f"{tbl.metadata.get('contextual_snippet_below')}"
                    )

            dl_payload = parsed_state.model_dump(mode="json")
            dl_scrubbed = json.dumps(_scrub_b64_for_json(dl_payload), indent=2, ensure_ascii=False)
            st.download_button(
                "Download JSON (sanitised thumbnails)",
                data=dl_scrubbed.encode("utf-8"),
                file_name=f"{parsed_state.filename or 'parsed'}.parsed.json",
                mime="application/json",
            )
