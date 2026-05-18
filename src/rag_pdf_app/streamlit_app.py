"""Streamlit entrypoint for environment checks and PDF ingestion lab."""

from __future__ import annotations

import base64
import json
import os
from typing import Any

import streamlit as st
from google.auth.exceptions import DefaultCredentialsError

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
            "Host: `gcloud auth application-default login`\n"
            "Docker Compose: mount host ADC (see docker-compose `GCP_ADC_HOST_PATH`) "
            "and GOOGLE_APPLICATION_CREDENTIALS, or mount a service-account JSON."
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
    overview_tab, parsing_tab, rag_tab = st.tabs(
        ["Overview", "PDF parsing", "Phase 1 · IFC RAG"]
    )

    with overview_tab:
        st.success("Loaded settings from environment.")
        st.json(
            {
                "GOOGLE_CLOUD_PROJECT": settings_obj.google_cloud_project,
                "GOOGLE_CLOUD_LOCATION": settings_obj.google_cloud_location,
                "GOOGLE_GENAI_USE_VERTEXAI": settings_obj.use_vertex_ai,
                "VERTEX_GENERATIVE_MODEL": settings_obj.vertex_generative_model,
                "VERTEX_EMBEDDING_MODEL": settings_obj.vertex_embedding_model,
                "RAG_QDRANT_COLLECTION": settings_obj.rag_qdrant_collection,
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
                    try:
                        out = generate_plain_text(prompt, settings_obj)
                    except DefaultCredentialsError as cred_exc:
                        st.error(
                            "Google credentials are not available **inside this process**. "
                            "On the host, run `gcloud auth application-default login`. "
                            "In **Docker**, mount that JSON and set "
                            "`GOOGLE_APPLICATION_CREDENTIALS` "
                            "(see `docker-compose.yml` for `GCP_ADC_HOST_PATH`)."
                        )
                        st.exception(cred_exc)
                    else:
                        st.write(out)

        st.markdown(
            "Phase 1 RAG lives in the **Phase 1 · IFC RAG** tab (ingest, dual retrieval, Gemini)."
        )

    with parsing_tab:
        from rag_pdf_app.parsing.pipeline import parse_pdf_bytes

        st.subheader("Structured PDF parsing")
        st.caption("Provide PDFs via **Upload** below (standard workflow).")

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

    with rag_tab:
        from rag_pdf_app.rag.embeddings import vertex_text_embeddings
        from rag_pdf_app.rag.ingest import ingest_pdf_bytes_to_indexes
        from rag_pdf_app.rag.query import run_phase1_rag
        from rag_pdf_app.rag.stores import load_faiss_index

        st.subheader("Baseline RAG — IFC Annual Report (text only)")
        st.markdown(
            "**Provide the PDF via upload** (normal flow). Ingest runs once (Vertex embeddings → "
            "**FAISS** on disk + **Qdrant**). Each question retrieves on **both** backends for "
            "latency/score comparison; Gemini answers use **FAISS** hits."
        )

        upload_rag = st.file_uploader(
            "Upload PDF",
            type=["pdf"],
            key="rag_pdf_upload",
            help="Preferred: choose the annual report (or any text PDF) from your machine.",
        )

        layout_chunks = st.checkbox("Structure-aware chunking (pdfminer)", value=True)
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Ingest & index", type="primary"):
                try:
                    if upload_rag is not None:
                        raw_pdf = upload_rag.getvalue()
                        name = upload_rag.name
                        with st.spinner("Embedding + FAISS + Qdrant…"):
                            outcome = ingest_pdf_bytes_to_indexes(
                                settings_obj,
                                raw_pdf,
                                name,
                                use_layout_chunking=layout_chunks,
                            )
                    else:
                        st.warning("Upload a PDF above to ingest.")
                        outcome = None
                    if outcome:
                        emb = vertex_text_embeddings(settings_obj)
                        st.session_state["phase1_faiss"] = load_faiss_index(emb, settings_obj)
                        st.session_state["phase1_ingest_meta"] = outcome
                        st.success(
                            f"Indexed **{outcome.chunk_count}** chunks · "
                            f"{outcome.embedding_dimensions}d · "
                            f"FAISS `{outcome.faiss_path}` · "
                            f"Qdrant `{outcome.qdrant_collection}`"
                        )
                        if outcome.notes:
                            with st.expander("Ingest notes"):
                                for n in outcome.notes:
                                    st.markdown(f"- `{n}`")
                except Exception as exc:  # noqa: BLE001 — surface stack in lab UI
                    st.error("Ingest failed (Vertex, Qdrant reachability, or empty PDF).")
                    st.exception(exc)

        with col_b:
            if st.button("Reload FAISS from disk"):
                try:
                    emb = vertex_text_embeddings(settings_obj)
                    st.session_state["phase1_faiss"] = load_faiss_index(emb, settings_obj)
                    st.success(
                        "Loaded FAISS index — ensure Qdrant already has the same collection."
                    )
                except Exception as exc:  # noqa: BLE001
                    st.error("Could not load FAISS store.")
                    st.exception(exc)

        faiss_store = st.session_state.get("phase1_faiss")
        if faiss_store is None:
            st.info("Upload a PDF and run **Ingest & index**, or reload FAISS from disk to query.")
        else:
            q_text = st.text_area("Question about the report", height=100)
            if st.button("Ask (retrieve + Gemini)", disabled=not q_text.strip()):
                with st.spinner("Retrieving + generating…"):
                    try:
                        result = run_phase1_rag(
                            settings_obj,
                            q_text.strip(),
                            faiss_store=faiss_store,
                        )
                    except Exception as exc:  # noqa: BLE001
                        st.error("RAG query failed.")
                        st.exception(exc)
                    else:
                        st.markdown("### Answer")
                        st.write(result.answer)
                        st.caption(
                            f"Langfuse trace: **{'yes' if result.langfuse_traced else 'no'}** "
                            "(needs `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY`)."
                        )

                        r = result.retrieval
                        c1, c2 = st.columns(2)
                        with c1:
                            st.metric("FAISS retrieval ms", f"{r.faiss_timing.latency_ms:.2f}")
                            st.caption(r.faiss_timing.metric)
                            for h in r.faiss_hits:
                                title = f"FAISS · {h.chunk_id[:12]}… · score {h.score:.4f}"
                                with st.expander(title):
                                    st.text(h.text[:2000])
                        with c2:
                            st.metric("Qdrant retrieval ms", f"{r.qdrant_timing.latency_ms:.2f}")
                            st.caption(r.qdrant_timing.metric)
                            for h in r.qdrant_hits:
                                with st.expander(
                                    f"Qdrant · {h.chunk_id[:12]}… · score {h.score:.4f}"
                                ):
                                    st.text(h.text[:2000])

                        with st.expander("FAISS vs Qdrant — when to use which"):
                            st.markdown(
                                """
**FAISS (local)** — Best for single-node prototypes, zero network hops after build,
simple files-on-disk deployment. Weaknesses: no multi-user ACLs, no incremental REST API,
you manage persistence and backups yourself.

**Qdrant (service)** — HTTP/gRPC API, horizontal scaling, filtering/payload queries,
multi-collection ops—better when several apps share vectors or you need managed ingestion.
Adds network latency vs purely local FAISS.

**This lab** keeps both in sync so you can compare raw query latency on identical embeddings;
scores differ because LangChain FAISS defaults to L2 distance while this Qdrant collection
uses cosine similarity.
"""
                            )

                        if r.notes:
                            with st.expander("Score interpretation"):
                                for line in r.notes:
                                    st.markdown(f"- {line}")
