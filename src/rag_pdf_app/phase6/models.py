"""Dataclasses for Phase 6 visual-patch pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class PatchPlacement:
    """Where a patch sits on the page raster."""

    page_index: int  # 0-based
    row_index: int
    col_index: int
    x0_px: int  # pixmap coordinates
    y0_px: int
    width_px: int
    height_px: int


@dataclass(frozen=True, slots=True)
class PatchRecord:
    """One cropped patch destined for multimodal embeddings."""

    patch_id: str
    pdf_sha256: str
    source_filename: str
    placement: PatchPlacement
    patch_png_bytes: bytes
    dpi: int
    pixmap_page_width: int  # raster width before tiling
    pixmap_page_height: int


@dataclass
class Phase6IngestOutcome:
    """Result of indexing all patches for one PDF."""

    pdf_sha256: str
    source_filename: str
    patch_count: int
    vector_dimension: int
    qdrant_collection: str
    embedding_model_name: str
    notes: list[str] = field(default_factory=list)


@dataclass()
class PatchRetrievalHit:
    """Ranked patch from Qdrant + optional rerank."""

    patch_id: str
    score: float
    payload: dict[str, Any]
    coarse_score: float | None = None  # cosine before rerank
