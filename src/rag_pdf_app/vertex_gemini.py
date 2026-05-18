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


def caption_document_image(
    settings: Settings,
    *,
    image_bytes: bytes,
    mime_type: str,
    neighbour_above: str | None,
    neighbour_below: str | None,
) -> str:
    """Multimodal caption for inline PDF figures routed through Vertex."""
    preamble = (
        "Annotate figures for retrieval. Summarise factual details in short sentences about "
        "objects, legends, handwriting, screenshots, diagrams, stamps, logos, charts."
    )

    parts: list[types.Part] = [types.Part.from_text(text=preamble)]
    if neighbour_above:
        parts.append(
            types.Part.from_text(
                text=f"Neighbour paragraph above the figure (may truncate):\n{neighbour_above}"
            )
        )
    if neighbour_below:
        parts.append(
            types.Part.from_text(
                text=f"Neighbour paragraph below the figure (may truncate):\n{neighbour_below}"
            )
        )
    parts.append(types.Part.from_bytes(data=image_bytes, mime_type=mime_type))

    client = client_for(settings)
    response = client.models.generate_content(
        model=settings.vertex_generative_model,
        contents=parts,
        config=types.GenerateContentConfig(max_output_tokens=768, temperature=0.25),
    )
    caption = getattr(response, "text", None)
    if caption:
        return str(caption).strip()
    raise RuntimeError("Gemini multimodal caption failed; inspect Vertex logs or IAM.")


def summarize_table_for_rag(
    settings: Settings,
    as_markdown: str | None,
    as_csv: str | None,
) -> str:
    """LLM condensation of Camelot/PyMuPDF table exports."""

    chunks: list[str] = []
    if as_markdown and as_markdown.strip():
        chunks.append(as_markdown.strip())
    if as_csv and as_csv.strip():
        chunks.append("CSV snapshot:\n" + as_csv.strip())
    blob = "\n\n".join(chunks).strip()
    if not blob:
        return ""
    prompt = (
        "Give ≤6 terse bullets capturing headings, quantitative facts, time spans, locales, "
        "counterparties,\nunits. Preserve numbers.\n\n" + blob[:12000]
    )
    return generate_plain_text(prompt, settings)
