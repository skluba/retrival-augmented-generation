"""Ingest raster PDF patches → multilingual CLIP embeddings → dedicated Qdrant collection."""

from __future__ import annotations

from typing import Any

from qdrant_client import QdrantClient

from rag_pdf_app.config import Settings
from rag_pdf_app.phase6.clip_embed import png_list_to_embeddings, sentence_transformers_clip_backend
from rag_pdf_app.phase6.models import PatchRecord, Phase6IngestOutcome
from rag_pdf_app.phase6.pdf_patches import iter_pdf_patch_records, sha256_pdf
from rag_pdf_app.phase6.visual_store import (
    payload_from_patch_record,
    reset_phase6_visual_collection,
    upsert_visual_patches_batch,
)


def ingest_phase6_visual_pdf(
    client: QdrantClient,
    settings: Settings,
    pdf_bytes: bytes,
    source_filename: str,
    *,
    device: str | None = None,
) -> Phase6IngestOutcome:
    """Rebuild the Phase 6 Qdrant collection (single-PDF lab semantics).

    Calling this replaces the Phase 6 collection contents so uploads stay deterministic.
    """

    notes: list[str] = []
    pdf_digest = sha256_pdf(pdf_bytes)
    model_key = settings.phase6_sentence_transformers_clip_model

    _model, embedding_dim = sentence_transformers_clip_backend(settings)
    reset_phase6_visual_collection(client, settings.phase6_qdrant_collection, embedding_dim)
    notes.append(f"phase6_reset_collection:{settings.phase6_qdrant_collection}:{embedding_dim}")

    batch_records: list[PatchRecord] = []
    vectors_acc: list[list[float]] = []
    payloads_acc: list[dict[str, Any]] = []
    ingest_batch = settings.phase6_ingest_encode_batch_size

    def flush() -> None:
        if not batch_records:
            return
        blobs = [r.patch_png_bytes for r in batch_records]
        mats, dim = png_list_to_embeddings(
            png_blobs=blobs,
            settings=settings,
            batch_size=settings.phase6_embedding_batch_size,
            device=device,
        )
        assert dim == embedding_dim

        payloads = [
            payload_from_patch_record(rec, embedding_model_name=model_key) for rec in batch_records
        ]
        payloads_acc.extend(payloads)
        vectors_acc.extend(mats)
        batch_records.clear()

    for rec in iter_pdf_patch_records(
        pdf_bytes,
        source_filename=source_filename,
        dpi=settings.phase6_render_dpi,
        patch_size_px=settings.phase6_patch_size_px,
        stride_px=settings.phase6_patch_stride_px,
        max_pages=settings.phase6_max_pages,
    ):
        batch_records.append(rec)
        if len(batch_records) >= ingest_batch:
            flush()

    flush()

    upsert_visual_patches_batch(
        client,
        collection=settings.phase6_qdrant_collection,
        vectors=vectors_acc,
        payloads=payloads_acc,
    )

    return Phase6IngestOutcome(
        pdf_sha256=pdf_digest,
        source_filename=source_filename,
        patch_count=len(vectors_acc),
        vector_dimension=embedding_dim,
        qdrant_collection=settings.phase6_qdrant_collection,
        embedding_model_name=model_key,
        notes=[
            *notes,
            f"phase6_visual_patches_written:{len(vectors_acc)}",
        ],
    )
