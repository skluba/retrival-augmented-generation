"""Vertex AI embeddings via LangChain."""

from __future__ import annotations

from langchain_google_vertexai import VertexAIEmbeddings

from rag_pdf_app.config import Settings


def vertex_text_embeddings(settings: Settings) -> VertexAIEmbeddings:
    return VertexAIEmbeddings(
        project=settings.google_cloud_project,
        location=settings.google_cloud_location,
        model_name=settings.vertex_embedding_model,
    )
