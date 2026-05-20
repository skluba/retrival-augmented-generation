"""Streamlit entrypoint for environment checks and PDF ingestion lab."""

from __future__ import annotations

import os

import streamlit as st
from google.auth.exceptions import DefaultCredentialsError

from rag_pdf_app import streamlit_ui_copy as uic
from rag_pdf_app.config import Settings, clear_settings_cache, get_settings

st.set_page_config(page_title=uic.PAGE_TITLE, layout="wide")
st.title(uic.TITLE)
st.caption(uic.CAPTION_MAIN)


with st.sidebar:
    st.markdown(uic.SIDEBAR_ENV_MARKDOWN_HEADING)
    st.code(uic.SIDEBAR_ADC_INSTRUCTIONS_TEXT, language="text")
    if st.button(uic.SIDEBAR_RELOAD_ENV_BUTTON):
        clear_settings_cache()
        st.rerun()

settings_obj: Settings | None = None

try:
    settings_obj = get_settings()
except Exception as exc:  # noqa: BLE001
    st.error(uic.ERR_SETTINGS_LOAD)
    st.exception(exc)
else:
    overview_tab, rag_tab, phase6_tab = st.tabs(list(uic.TAB_LABELS))

    with overview_tab:
        st.success(uic.OVERVIEW_SUCCESS)
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

        if st.checkbox(uic.OVERVIEW_SMOKE_CHECKBOX_LABEL):
            from rag_pdf_app.vertex_gemini import generate_plain_text

            prompt = st.text_area(uic.OVERVIEW_PROMPT_LABEL, value=uic.OVERVIEW_PROMPT_DEFAULT)
            if st.button(uic.OVERVIEW_GEMINI_RUN_BUTTON):
                with st.spinner(uic.OVERVIEW_GEMINI_SPINNER):
                    try:
                        out = generate_plain_text(prompt, settings_obj)
                    except DefaultCredentialsError as cred_exc:
                        st.error(uic.OVERVIEW_GEMINI_CREDENTIAL_ERROR)
                        st.exception(cred_exc)
                    else:
                        st.write(out)

        st.markdown(uic.OVERVIEW_FOOTNOTE_MARKDOWN)

    with rag_tab:
        from rag_pdf_app.rag.embeddings import vertex_text_embeddings
        from rag_pdf_app.rag.ingest import ingest_pdf_bytes_to_indexes
        from rag_pdf_app.rag.query import run_phase1_rag
        from rag_pdf_app.rag.stores import load_faiss_index
        from rag_pdf_app.rag.table_plot import chartable_numeric_frame, dataframe_from_hits

        st.subheader(uic.RAG_SUBHEADER)
        st.markdown(uic.RAG_INTRO_MARKDOWN)

        upload_rag = st.file_uploader(
            uic.RAG_UPLOAD_LABEL,
            type=["pdf"],
            key="rag_pdf_upload",
            help=uic.RAG_UPLOAD_HELP,
        )

        layout_chunks = st.checkbox(uic.RAG_LAYOUT_CHUNKS_CHECKBOX, value=True)
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button(uic.RAG_INGEST_BUTTON, type="primary"):
                try:
                    if upload_rag is not None:
                        raw_pdf = upload_rag.getvalue()
                        name = upload_rag.name
                        with st.spinner(uic.RAG_INGEST_SPINNER):
                            outcome = ingest_pdf_bytes_to_indexes(
                                settings_obj,
                                raw_pdf,
                                name,
                                use_layout_chunking=layout_chunks,
                            )
                    else:
                        st.warning(uic.RAG_INGEST_UPLOAD_WARN)
                        outcome = None
                    if outcome:
                        emb = vertex_text_embeddings(settings_obj)
                        st.session_state["phase1_faiss"] = load_faiss_index(emb, settings_obj)
                        st.session_state["phase1_ingest_meta"] = outcome
                        st.success(
                            uic.RAG_INGEST_SUCCESS_TEMPLATE.format(
                                chunk_count=outcome.chunk_count,
                                embedding_dimensions=outcome.embedding_dimensions,
                                faiss_path=outcome.faiss_path,
                                qdrant_collection=outcome.qdrant_collection,
                            )
                        )
                        if outcome.notes:
                            with st.expander(uic.RAG_EXPANDER_INGEST_NOTES):
                                for n in outcome.notes:
                                    st.markdown(f"- `{n}`")
                except Exception as exc:  # noqa: BLE001 — surface stack in lab UI
                    st.error(uic.RAG_ERR_INGEST)
                    st.exception(exc)

        with col_b:
            if st.button(uic.RAG_RELOAD_FAISS_BUTTON):
                try:
                    emb = vertex_text_embeddings(settings_obj)
                    st.session_state["phase1_faiss"] = load_faiss_index(emb, settings_obj)
                    st.success(uic.RAG_RELOAD_FAISS_SUCCESS)
                except Exception as exc:  # noqa: BLE001
                    st.error(uic.RAG_ERR_FAISS_LOAD)
                    st.exception(exc)

        faiss_store = st.session_state.get("phase1_faiss")
        if faiss_store is None:
            st.info(uic.RAG_INFO_UPLOAD_OR_RELOAD_FAISS)
        else:
            q_text = st.text_area(uic.RAG_QUESTION_LABEL, height=100)
            if st.button(uic.RAG_ASK_BUTTON, disabled=not q_text.strip()):
                with st.spinner(uic.RAG_QUERY_SPINNER):
                    try:
                        result = run_phase1_rag(
                            settings_obj,
                            q_text.strip(),
                            faiss_store=faiss_store,
                        )
                    except Exception as exc:  # noqa: BLE001
                        st.error(uic.RAG_ERR_QUERY)
                        st.exception(exc)
                    else:
                        st.markdown(uic.RAG_MARKDOWN_HEADING_ANSWER)
                        st.write(result.answer)
                        if not result.semantic_cache_hit and settings_obj.rag_plotting_enabled:
                            df_plot, plot_note = dataframe_from_hits(
                                result.retrieval.faiss_hits,
                                settings_obj,
                            )
                            if df_plot is not None:
                                st.subheader(uic.RAG_CHART_SUBHEADER)
                                st.dataframe(df_plot.head(80), use_container_width=True)
                                num_df = chartable_numeric_frame(df_plot)
                                if not num_df.empty:
                                    st.bar_chart(num_df.head(40))
                                else:
                                    st.caption(uic.RAG_CAPTION_CHART_SKIP_NUMERIC)
                            else:
                                st.caption(
                                    uic.RAG_CAPTION_NO_CSV_PREVIEW_TEMPLATE.format(plot_note=plot_note)
                                )
                        if result.semantic_cache_hit:
                            st.success(uic.RAG_SUCCESS_SEMANTIC_CACHE)
                            if result.semantic_cache_similarity is not None:
                                sim = result.semantic_cache_similarity
                                st.caption(uic.RAG_CAPTION_SEMANTIC_SIM_TEMPLATE.format(sim=sim))
                        if result.multi_hop_used:
                            st.info(uic.RAG_INFO_MULTI_HOP)
                        traced = (
                            uic.RAG_LANGFUSE_YES
                            if result.langfuse_traced
                            else uic.RAG_LANGFUSE_NO
                        )
                        st.caption(uic.RAG_CAPTION_LANGFUSE_TEMPLATE.format(traced_label=traced))

                        r = result.retrieval
                        c1, c2 = st.columns(2)
                        with c1:
                            st.metric(
                                uic.RAG_METRIC_FAISS_LABEL,
                                f"{r.faiss_timing.latency_ms:.2f}",
                            )
                            st.caption(r.faiss_timing.metric)
                            for h in r.faiss_hits:
                                kind = h.metadata.get("chunk_kind", uic.DEFAULT_CHUNK_KIND)
                                title = uic.RAG_FAISS_HIT_EXPANDER_TEMPLATE.format(
                                    kind=kind,
                                    chunk_prefix=uic.rag_faiss_hit_chunk_prefix(h.chunk_id),
                                    score=h.score,
                                )
                                with st.expander(title):
                                    st.text(h.text[: uic.RAG_HIT_TEXT_PREVIEW_CHARS])
                        with c2:
                            st.metric(
                                uic.RAG_METRIC_QDRANT_LABEL,
                                f"{r.qdrant_timing.latency_ms:.2f}",
                            )
                            st.caption(r.qdrant_timing.metric)
                            for h in r.qdrant_hits:
                                title = uic.RAG_QDRANT_HIT_EXPANDER_TEMPLATE.format(
                                    chunk_prefix=uic.rag_faiss_hit_chunk_prefix(h.chunk_id),
                                    score=h.score,
                                )
                                with st.expander(title):
                                    st.text(h.text[: uic.RAG_HIT_TEXT_PREVIEW_CHARS])

                        with st.expander(uic.RAG_EXPANDER_FAISS_VS_QDRANT):
                            st.markdown(uic.RAG_MARKDOWN_FAISS_VS_QDRANT)

                        if r.notes:
                            with st.expander(uic.RAG_EXPANDER_SCORE_INTERPRETATION):
                                # Plain text: dual_retrieval_notes can contain LLM-derived strings;
                                # avoid st.markdown so document/markdown injection cannot run here.
                                st.code("\n".join(r.notes), language=None)

    with phase6_tab:
        st.subheader(uic.PHASE6_SUBHEADER)
        st.markdown(uic.PHASE6_INTRO_MARKDOWN)
        miss = uic.phase6_optional_deps_hint()
        if miss:
            st.warning(miss + uic.PHASE6_RELOAD_AFTER_EXTRA)
            st.markdown(uic.PHASE6_ENV_MARKDOWN_FALLBACK)
        else:
            import io as _phase6_io

            from rag_pdf_app.phase6.coords import patch_placement_from_payload, rerender_patch_png
            from rag_pdf_app.phase6.ingest_visual import ingest_phase6_visual_pdf
            from rag_pdf_app.phase6.retrieve_visual import retrieve_phase6_visual_patches
            from rag_pdf_app.rag.stores import qdrant_client

            st.caption(
                uic.PHASE6_QDRANT_CAPTION_TEMPLATE.format(
                    collection=settings_obj.phase6_qdrant_collection,
                )
            )

            upload_vis = st.file_uploader(
                uic.PHASE6_UPLOAD_LABEL,
                type=["pdf"],
                key="phase6_pdf_upload",
            )

            ingest_col_left, ingest_col_right = st.columns(2)
            with ingest_col_left:
                if st.button(uic.PHASE6_INGEST_BUTTON, type="primary", key="p6_ingest"):
                    if upload_vis is None:
                        st.warning(uic.PHASE6_WARN_UPLOAD_FOR_INGEST)
                    else:
                        raw_vis = upload_vis.getvalue()
                        name_vis = upload_vis.name
                        with st.spinner(uic.PHASE6_INGEST_SPINNER):
                            try:
                                qdr = qdrant_client(settings_obj)
                                outcome = ingest_phase6_visual_pdf(
                                    qdr, settings_obj, raw_vis, name_vis
                                )
                            except Exception as exc:  # noqa: BLE001
                                st.error(uic.PHASE6_ERR_INGEST)
                                st.exception(exc)
                            else:
                                st.session_state["phase6_pdf_bytes"] = raw_vis
                                st.session_state["phase6_pdf_sha256"] = outcome.pdf_sha256
                                st.session_state["phase6_source_name"] = name_vis
                                st.session_state["phase6_ingest"] = outcome
                                st.success(
                                    uic.PHASE6_INGEST_SUCCESS_TEMPLATE.format(
                                        patch_count=outcome.patch_count,
                                        vector_dimension=outcome.vector_dimension,
                                        collection=outcome.qdrant_collection,
                                        embedding_model=outcome.embedding_model_name,
                                    )
                                )
                                if outcome.notes:
                                    with st.expander(uic.RAG_EXPANDER_INGEST_NOTES):
                                        st.code("\n".join(outcome.notes), language=None)

            with ingest_col_right:
                if st.button(uic.PHASE6_FORGET_SESSION_BUTTON, key="p6_clear"):
                    for key in ("phase6_pdf_bytes", "phase6_pdf_sha256", "phase6_source_name"):
                        st.session_state.pop(key, None)
                    st.success(uic.PHASE6_SESSION_CLEARED)

            if st.session_state.get("phase6_pdf_bytes"):
                digest = str(st.session_state.get("phase6_pdf_sha256", ""))
                digest_disp = digest[:18] + "…" if len(digest) > 18 else (digest or "?")
                raw_name = str(
                    st.session_state.get("phase6_source_name")
                    or uic.PHASE6_UPLOAD_DEFAULT_FILENAME
                )
                basename_safe = (
                    os.path.basename(raw_name) or raw_name or uic.PHASE6_UPLOAD_DEFAULT_FILENAME
                )
                # Filenames come from browser uploads — never interpolate into Markdown widgets.
                st.caption(uic.PHASE6_CAPTION_SESSION_READY)
                st.text(
                    uic.PHASE6_SESSION_PLAINTEXT_TEMPLATE.format(
                        basename=basename_safe,
                        digest_disp=digest_disp,
                    )
                )

            q_visual = st.text_area(
                uic.PHASE6_QUESTION_LABEL,
                height=96,
                key="phase6_question",
            )
            rerender_dpi_slider = st.slider(
                uic.PHASE6_SLIDER_RERENDER_DPI_LABEL,
                min_value=int(settings_obj.phase6_render_dpi),
                max_value=min(216, max(int(settings_obj.phase6_render_dpi), 216)),
                value=int(settings_obj.phase6_render_dpi),
                help=uic.PHASE6_SLIDER_RERENDER_DPI_HELP,
            )

            if st.button(uic.PHASE6_ASK_BUTTON, key="p6_ask"):
                pdf_buf_local = st.session_state.get("phase6_pdf_bytes")
                pdf_digest = str(st.session_state.get("phase6_pdf_sha256") or "").strip()
                if not pdf_buf_local:
                    st.warning(uic.PHASE6_WARN_INGEST_FIRST)
                elif not pdf_digest:
                    st.warning(uic.PHASE6_WARN_NO_DIGEST)
                elif not q_visual.strip():
                    st.warning(uic.PHASE6_WARN_EMPTY_QUESTION)
                else:
                    from rag_pdf_app.vertex_gemini import generate_visual_rag_answer_from_patches

                    qdr_local = qdrant_client(settings_obj)
                    hits_local = []
                    telem_local: dict[str, str] = {}
                    thumbs: list[bytes] = []
                    try:
                        with st.spinner(uic.PHASE6_SPINNER_CLIP_RETRIEVE):
                            hits_local, telem_local = retrieve_phase6_visual_patches(
                                qdr_local,
                                settings_obj,
                                query=q_visual.strip(),
                                pdf_sha256=pdf_digest,
                                device=None,
                            )
                            thumbs = []
                            for hit in hits_local:
                                pay = hit.payload
                                thumbs.append(
                                    rerender_patch_png(
                                        pdf_buf_local,
                                        patch_page_index=int(pay["page_index"]),
                                        placement=patch_placement_from_payload(pay),
                                        pixmap_page_width=int(pay["pixmap_page_width"]),
                                        pixmap_page_height=int(pay["pixmap_page_height"]),
                                        dpi=float(rerender_dpi_slider),
                                    )
                                )
                        if telem_local:
                            with st.expander(uic.PHASE6_EXPANDER_RETRIEVAL_TELEMETRY):
                                st.json(telem_local)

                        if not hits_local:
                            st.warning(uic.PHASE6_WARN_NO_PATCHES)
                        else:
                            with st.spinner(uic.PHASE6_SPINNER_GEMINI):
                                labelled = [(f"[P{i}]", b) for i, b in enumerate(thumbs, start=1)]
                                answer_txt = generate_visual_rag_answer_from_patches(
                                    settings_obj,
                                    user_query=q_visual.strip(),
                                    labelled_patch_pngs=labelled,
                                )

                            st.markdown(uic.PHASE6_MARKDOWN_HEADING_ANSWER)
                            st.write(answer_txt)

                            st.markdown(uic.PHASE6_MARKDOWN_HEADING_SOURCES)
                            for idx, png in enumerate(thumbs, start=1):
                                hit = hits_local[idx - 1]
                                pay = hit.payload
                                pg = int(pay.get("page_index", 0)) + 1
                                r_ix = pay.get("row_index")
                                c_ix = pay.get("col_index")
                                cap = uic.PHASE6_ATTR_LINE_PREFIX_TEMPLATE.format(
                                    idx=idx,
                                    page=pg,
                                    r_ix=r_ix,
                                    c_ix=c_ix,
                                )
                                if settings_obj.phase6_visual_maxsim_rerank and (
                                    hit.coarse_score is not None
                                    and abs(hit.coarse_score - hit.score) > 1e-6
                                ):
                                    cap += uic.PHASE6_ATTR_SCORE_RERANK_TEMPLATE.format(
                                        coarse=hit.coarse_score,
                                        score=hit.score,
                                    )
                                else:
                                    cap += uic.PHASE6_ATTR_SCORE_SIMPLE_TEMPLATE.format(
                                        score=hit.score,
                                    )
                                st.markdown(cap)
                                st.image(
                                    _phase6_io.BytesIO(png),
                                    caption=None,
                                    use_container_width=True,
                                )

                            with st.expander(uic.PHASE6_EXPANDER_COMPARE_PHASE1):
                                st.markdown(uic.PHASE6_COMPARE_MARKDOWN)
                    except Exception as exc:  # noqa: BLE001
                        st.error(uic.PHASE6_ERR_QUERY)
                        st.exception(exc)
