"""Streamlit entrypoint for environment checks and PDF ingestion lab."""

from __future__ import annotations

import os

import streamlit as st
from google.auth.exceptions import DefaultCredentialsError

from rag_pdf_app.config import Settings, clear_settings_cache, get_settings

st.set_page_config(page_title="PDF RAG Lab", layout="wide")
st.title("PDF RAG workspace")
st.caption("Gemini · Vertex AI · hybrid retrieval (FAISS + Qdrant) · tables · figure captions")


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
    overview_tab, rag_tab = st.tabs(["Overview", "Ingest & query (RAG)"])

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
            "Upload a PDF, index **FAISS + Qdrant**, and ask questions in "
            "**Ingest & query (RAG)** — hybrid retrieval (dense + BM25), structured **table** "
            "chunks, **figure** caption chunks, optional semantic cache and multi-hop "
            "(see `.env`)."
        )

    with rag_tab:
        from rag_pdf_app.rag.embeddings import vertex_text_embeddings
        from rag_pdf_app.rag.ingest import ingest_pdf_bytes_to_indexes
        from rag_pdf_app.rag.query import run_phase1_rag
        from rag_pdf_app.rag.stores import load_faiss_index
        from rag_pdf_app.rag.table_plot import chartable_numeric_frame, dataframe_from_hits

        st.subheader("Baseline RAG — text, tables (5.1), and figure captions (5.2)")
        st.markdown(
            "**Provide the PDF via upload** (normal flow). Ingest runs once (Vertex embeddings → "
            "**FAISS** on disk + **Qdrant**). **Phase 5.1** adds **table chunks**; **Phase 5.2** "
            "adds **figure / chart chunks** from Gemini image captions plus nearby PDF text. "
            "By default (`RAG_IMAGE_REQUIRE_FIGURE_LABEL_NEARBY`, see `.env`) only rasters with a "
            "nearby **`Figure N` / `Fig. N`** line are indexed—fewer bogus “images” beside plain "
            "headings. "
            "Each question retrieves on **both** backends; Gemini answers use **FAISS** hits. "
            "\n\n**Phase 4 (`.env`, on by default):** `RAG_SEMANTIC_CACHE_ENABLED` reuses answers "
            "for similar questions (skipped when using page-window filters). "
            "`RAG_MULTI_HOP_ENABLED` runs a second retrieval pass after an LLM-suggested query. "
            "Set either to `false` to disable."
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
                        if not result.semantic_cache_hit and settings_obj.rag_plotting_enabled:
                            df_plot, plot_note = dataframe_from_hits(
                                result.retrieval.faiss_hits,
                                settings_obj,
                            )
                            if df_plot is not None:
                                st.subheader("Chart · retrieved table preview")
                                st.dataframe(df_plot.head(80), use_container_width=True)
                                num_df = chartable_numeric_frame(df_plot)
                                if not num_df.empty:
                                    st.bar_chart(num_df.head(40))
                                else:
                                    st.caption(
                                        "Numeric chart skipped — no numeric columns detected "
                                        "(table still shown above)."
                                    )
                            else:
                                st.caption(
                                    "Plotting: no CSV-backed table in top FAISS hits "
                                    f"({plot_note})."
                                )
                        if result.semantic_cache_hit:
                            st.success(
                                "Semantic cache hit — similar prior query (see retrieval notes)."
                            )
                            if result.semantic_cache_similarity is not None:
                                sim = result.semantic_cache_similarity
                                st.caption(f"Cache cosine similarity ≈ **{sim:.3f}**")
                        if result.multi_hop_used:
                            st.info("Multi-hop retrieval merged a second FAISS pass into context.")
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
                                kind = h.metadata.get("chunk_kind", "narrative")
                                title = f"FAISS · {kind} · {h.chunk_id[:12]}… · score {h.score:.4f}"
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
                                # Plain text: dual_retrieval_notes can contain LLM-derived strings;
                                # avoid st.markdown so document/markdown injection cannot run here.
                                st.code("\n".join(r.notes), language=None)
