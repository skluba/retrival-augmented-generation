"""Multilingual CLIP embeddings via sentence-transformers (optional Phase 6 extra)."""

from __future__ import annotations

import io
from typing import Any

from PIL import Image

from rag_pdf_app.config import Settings

_model_cache: dict[str, Any] = {}
"""Process-local cache keyed by HF model identifier."""


def require_sentence_transformers() -> tuple[type[Any], Any]:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - optional deps
        msg = (
            "sentence-transformers is required for Phase 6 CLIP ingestion. Install with "
            "`uv sync --extra phase6` (brings Torch + multilingual CLIP weights)."
        )
        raise RuntimeError(msg) from exc
    import torch

    return SentenceTransformer, torch


def sentence_transformers_clip_backend(settings: Settings) -> tuple[Any, int]:
    """Return ``(SentenceTransformer model, embedding_dim)``."""

    name = settings.phase6_sentence_transformers_clip_model
    if name in _model_cache:
        m = _model_cache[name]
        dim = getattr(m, "get_sentence_embedding_dimension", lambda: 512)()
        return m, dim

    transformer_cls, _torch_module = require_sentence_transformers()
    model = transformer_cls(name)
    dim = model.get_sentence_embedding_dimension()
    _model_cache[name] = model
    return model, dim


def png_list_to_embeddings(
    *,
    png_blobs: list[bytes],
    settings: Settings,
    batch_size: int,
    device: str | None = None,
) -> tuple[list[list[float]], int]:
    """Encode patch PNG payloads as normalised cosine-ready vectors."""

    model, dim = sentence_transformers_clip_backend(settings)

    imgs: list[Image.Image] = []
    for blob in png_blobs:
        imgs.append(Image.open(io.BytesIO(blob)).convert("RGB"))

    out: list[list[float]] = []
    for i in range(0, len(imgs), batch_size):
        chunk = imgs[i : i + batch_size]
        vecs = model.encode(
            chunk,
            batch_size=min(batch_size, len(chunk)),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            device=device,
        )
        for row in vecs.tolist():  # type: ignore[attr-defined]
            out.append([float(v) for v in row])

    assert len(out) == len(imgs)

    return out, dim


def encode_queries_clip(
    *, texts: list[str], settings: Settings, device: str | None = None
) -> list[list[float]]:
    """Map text queries through the CLIP tower (normalised)."""

    model, _dim = sentence_transformers_clip_backend(settings)
    if not texts:
        return []

    bs = min(settings.phase6_embedding_batch_size, max(1, len(texts)))

    vec = model.encode(
        texts,
        batch_size=bs,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
        device=device,
    )
    rows = vec.tolist()  # type: ignore[attr-defined]
    return [[float(v) for v in row] for row in rows]
