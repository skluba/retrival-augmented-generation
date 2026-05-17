"""Thin wrapper around google-genai with Vertex AI (Application Default Credentials)."""

from __future__ import annotations

from google import genai
from google.genai import types

from rag_pdf_app.config import Settings


def client_for(settings: Settings) -> genai.Client:
    """Create a ``google-genai`` client routed through Vertex AI (no API keys)."""
    return genai.Client(
        vertexai=settings.use_vertex_ai,
        project=settings.google_cloud_project,
        location=settings.google_cloud_location,
    )


def generate_plain_text(prompt: str, settings: Settings) -> str:
    """Minimal text generation helper for scaffolding and smoke checks."""
    client = client_for(settings)
    response = client.models.generate_content(
        model=settings.vertex_generative_model,
        contents=prompt,
        config=types.GenerateContentConfig(max_output_tokens=512, temperature=0.2),
    )
    text = getattr(response, "text", None)
    if text:
        return str(text).strip()
    raise RuntimeError("Gemini returned no text; check quotas, IAM, or model availability.")
