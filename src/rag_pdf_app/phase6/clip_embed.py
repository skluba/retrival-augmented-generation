"""CLIP-space embeddings for Phase 6 (multilingual queries + HF vision tower).

``sentence-transformers/clip-ViT-B-32-multilingual-v1`` maps **natural-language queries** into
CLIP text space via a DistilBERT + dense stack; it ships **without** a ViT. Sentence Transformers
v5+ therefore lists this checkpoint as **text-only**, and handing :class:`~PIL.Image.Image` blobs to
:class:`sentence_transformers.SentenceTransformer` raises a modality error.

Patch PNG crops are encoded with a standard Hugging Face :class:`~transformers.CLIPModel`
(:attr:`phase6_clip_image_encoder_model`, default ViT-B/32 paired with multilingual text retrieval).
Queries keep using Sentence Transformer on :attr:`phase6_sentence_transformers_clip_model`.

Both halves must expose the **same embedding dimension** (defaults: 512 + 512).
"""

from __future__ import annotations

import io
from typing import Any

from PIL import Image

from rag_pdf_app.config import Settings

_text_model_cache: dict[str, Any] = {}
"""Process-local Sentence Transformer cache keyed by HF model identifier."""

_IMAGE_CLIP_KEYS: dict[tuple[str, str], tuple[Any, Any, int]] = {}
"""``(CLIPModel, CLIPProcessor, projection_dim)`` keyed by *(model id, torch device string)*."""

_ST_DIM_CACHE: dict[str, int] = {}
"""SentenceTransformer output dim per model id (queried lazily once)."""


def require_torch_module() -> Any:
    try:
        import torch

        return torch
    except ImportError as exc:  # pragma: no cover - optional deps
        msg = (
            "PyTorch is required for Phase 6 CLIP ingestion. Install with `uv sync --extra phase6`."
        )
        raise RuntimeError(msg) from exc


def require_sentence_transformers_cls() -> type[Any]:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - optional deps
        msg = (
            "sentence-transformers is required for Phase 6 multilingual text queries. "
            "Install with `uv sync --extra phase6`."
        )
        raise RuntimeError(msg) from exc
    return SentenceTransformer


def _torch_device(pref: str | None) -> Any:
    torch = require_torch_module()
    if pref:
        return torch.device(pref)
    if torch.cuda.is_available():  # type: ignore[union-attr]
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():  # type: ignore[union-attr]
        return torch.device("mps")
    return torch.device("cpu")


def _tensor_from_clip_image_features(torch_mod: Any, raw: Any) -> Any:
    """Normalize CLIP vision outputs across transformers APIs.

    v4 typically returns ``FloatTensor``. v5+ ``CLIPModel.get_image_features`` may return a
    :class:`~transformers.modeling_outputs.BaseModelOutputWithPooling` whose projected embeddings
    sit in ``pooler_output``.
    """

    if isinstance(raw, torch_mod.Tensor):
        return raw
    pooled = getattr(raw, "pooler_output", None)
    if isinstance(pooled, torch_mod.Tensor):
        return pooled
    msg = (
        f"Unexpected CLIP image feature output ({type(raw).__name__}); "
        "expected Tensor or ModelOutput.pooler_output."
    )
    raise TypeError(msg)


def phase6_hf_image_clip_bundle(
    settings: Settings,
    *,
    device: str | None = None,
) -> tuple[Any, Any, int]:
    """Return ``(CLIPModel, CLIPProcessor, projection_dim)`` on the resolved device."""

    try:
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as exc:  # pragma: no cover - transitive from sentence-transformers
        msg = "`transformers` is required for Phase 6 image encoding (CLIP vision tower)."
        raise RuntimeError(msg) from exc

    mid = settings.phase6_clip_image_encoder_model
    dev = _torch_device(device)
    key = (mid, str(dev))
    cached = _IMAGE_CLIP_KEYS.get(key)
    if cached is not None:
        return cached

    processor = CLIPProcessor.from_pretrained(mid)
    clip = CLIPModel.from_pretrained(mid)
    clip = clip.to(dev)
    clip.eval()

    proj = int(getattr(clip.config, "projection_dim", None) or 0)
    if proj < 1:
        msg = f"CLIP checkpoint {mid} has no usable projection_dim in config."
        raise ValueError(msg)

    bundle = (clip, processor, proj)
    _IMAGE_CLIP_KEYS[key] = bundle
    return bundle


def phase6_multilingual_text_sentence_transformer(
    settings: Settings,
) -> tuple[Any, int]:
    """Return ``(SentenceTransformer, embedding_dim)`` for CLIP-aligned **text**."""

    transformer_cls = require_sentence_transformers_cls()
    name = settings.phase6_sentence_transformers_clip_model
    if name in _text_model_cache:
        m = _text_model_cache[name]
        dim = _ST_DIM_CACHE.get(name)
        if dim is None:
            dim = m.get_sentence_embedding_dimension()
            _ST_DIM_CACHE[name] = int(dim)
        return m, int(dim)

    model = transformer_cls(name)
    dim = int(model.get_sentence_embedding_dimension())
    _text_model_cache[name] = model
    _ST_DIM_CACHE[name] = dim
    return model, dim


def ensure_phase6_clip_dims_aligned(
    settings: Settings,
    *,
    device: str | None = None,
) -> int:
    """Load both halves once and enforce matching vector width."""

    _, tdim = phase6_multilingual_text_sentence_transformer(settings)
    _, _, idim = phase6_hf_image_clip_bundle(settings, device=device)
    if tdim != idim:
        msg = (
            "Phase 6 text vs image embedding dimension mismatch "
            f"({tdim} from {settings.phase6_sentence_transformers_clip_model} vs "
            f"{idim} from {settings.phase6_clip_image_encoder_model}). "
            "Pick a CLIP ViT backbone whose projection_dim matches the multilingual encoder."
        )
        raise ValueError(msg)
    return tdim


def png_list_to_embeddings(
    *,
    png_blobs: list[bytes],
    settings: Settings,
    batch_size: int,
    device: str | None = None,
) -> tuple[list[list[float]], int]:
    """Encode patch PNG payloads as normalised cosine-ready vectors (HF CLIP vision tower)."""

    torch = require_torch_module()
    clip, processor, dim = phase6_hf_image_clip_bundle(settings, device=device)

    imgs: list[Image.Image] = []
    for blob in png_blobs:
        imgs.append(Image.open(io.BytesIO(blob)).convert("RGB"))

    dev = next(clip.parameters()).device

    out: list[list[float]] = []
    for i in range(0, len(imgs), batch_size):
        chunk = imgs[i : i + batch_size]
        pil_batch = processor(images=chunk, return_tensors="pt")
        pix = pil_batch["pixel_values"].to(dev)

        with torch.no_grad():  # type: ignore[union-attr]
            raw_feats = clip.get_image_features(pixel_values=pix)
        feats = _tensor_from_clip_image_features(torch, raw_feats)

        feats = feats / feats.norm(dim=-1, keepdim=True)
        nest = feats.detach().cpu().float().tolist()
        for row in nest:
            out.append([float(v) for v in row])

    assert len(out) == len(imgs)
    return out, dim


def encode_queries_clip(
    *, texts: list[str], settings: Settings, device: str | None = None
) -> list[list[float]]:
    """Map text queries through the multilingual CLIP-aligned text encoder."""

    model, _dim = phase6_multilingual_text_sentence_transformer(settings)
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
