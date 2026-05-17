"""Streamlit entrypoint: environment smoke check and scaffold for PDF RAG."""

from __future__ import annotations

import os

import streamlit as st

from rag_pdf_app.config import Settings, clear_settings_cache, get_settings

st.set_page_config(page_title="PDF RAG Lab", layout="wide")
st.title("PDF RAG workspace")
st.caption("Gemini 2.0 Flash via Vertex AI, LangChain, FAISS, Qdrant, Langfuse/RAGAS (scaffold)")


settings: Settings | None = None

with st.sidebar:
    st.markdown("### Environment")
    st.code(
        (
            "Use GCP Application Default Credentials (ADC)\n"
            "— no Gemini API keys. Example:\n"
            "`gcloud auth application-default login`\n\n"
            "For Docker mount your ADC JSON and set:\n"
            "`GOOGLE_APPLICATION_CREDENTIALS`"
        ),
        language="text",
    )
    if st.button("Reload `.env` / env"):
        clear_settings_cache()
        st.rerun()

try:
    settings = get_settings()
except Exception as exc:  # noqa: BLE001 — surface misconfiguration clearly in UI
    st.warning("Settings could not load. Set GCP / local env vars (see README).")
    st.exception(exc)
else:
    st.success("Loaded settings from environment.")
    st.json(
        {
            "GOOGLE_CLOUD_PROJECT": settings.google_cloud_project,
            "GOOGLE_CLOUD_LOCATION": settings.google_cloud_location,
            "GOOGLE_GENAI_USE_VERTEXAI": settings.use_vertex_ai,
            "VERTEX_GENERATIVE_MODEL": settings.vertex_generative_model,
            "QDRANT_URL": settings.qdrant_url,
            "LANGFUSE_HOST": settings.langfuse_host,
            "LANGFUSE_PUBLIC_KEY_set": settings.langfuse_public_key is not None,
            "LANGFUSE_SECRET_KEY_set": settings.langfuse_secret_key is not None,
            "GOOGLE_APPLICATION_CREDENTIALS_set": bool(os.getenv("GOOGLE_APPLICATION_CREDENTIALS")),
        }
    )

    if st.checkbox("Smoke test Gemini (calls Vertex AI; incurs quota)"):
        from rag_pdf_app.vertex_gemini import generate_plain_text

        prompt = st.text_area("Prompt", value="Respond with exactly: pong")
        if st.button("Run"):
            with st.spinner("Calling Gemini…"):
                out = generate_plain_text(prompt, settings)
            st.write(out)


st.markdown(
    "---\n**Next steps:** ingest PDFs (Docling / PyMuPDF / multimodal), "
    "chunk + embed into FAISS or Qdrant, retrieve with LangChain, trace with Langfuse, "
    "evaluate with RAGAS."
)
