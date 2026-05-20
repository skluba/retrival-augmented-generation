"""Qdrant IO for multilingual CLIP patch vectors."""

from __future__ import annotations

from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from rag_pdf_app.phase6.models import PatchRecord
from rag_pdf_app.rag.stores import reset_qdrant_collection, stable_point_uuid


def patch_point_id(patch_id: str) -> str:
    return stable_point_uuid(patch_id)


def reset_phase6_visual_collection(client: QdrantClient, collection: str, vector_size: int) -> None:
    reset_qdrant_collection(client, collection, vector_size)


def payload_from_patch_record(record: PatchRecord, *, embedding_model_name: str) -> dict[str, Any]:
    p = record.placement
    pdf_sha256 = record.pdf_sha256
    summary = (
        f"CLIP patch · page {p.page_index + 1} · r{p.row_index}c{p.col_index} · "
        f"{record.source_filename}"
    )

    meta: dict[str, Any] = {
        "patch_id": record.patch_id,
        "chunk_id": record.patch_id,
        "pdf_sha256": pdf_sha256,
        "source_filename": record.source_filename,
        "page_index": p.page_index,
        "row_index": p.row_index,
        "col_index": p.col_index,
        "x0_px": p.x0_px,
        "y0_px": p.y0_px,
        "width_px": p.width_px,
        "height_px": p.height_px,
        "dpi": record.dpi,
        "pixmap_page_width": record.pixmap_page_width,
        "pixmap_page_height": record.pixmap_page_height,
        "embedding_model_name": embedding_model_name,
        "text": summary,
        "embedding_model": embedding_model_name,
    }
    return meta


def upsert_visual_patches_batch(
    client: QdrantClient,
    *,
    collection: str,
    vectors: list[list[float]],
    payloads: list[dict[str, Any]],
    batch_size: int = 128,
) -> None:
    if len(vectors) != len(payloads):
        msg = "vectors/payloads length mismatch"
        raise ValueError(msg)

    points: list[qmodels.PointStruct] = []
    for vec, payload in zip(vectors, payloads, strict=True):
        cid = str(payload["patch_id"])
        points.append(
            qmodels.PointStruct(
                id=patch_point_id(cid),
                vector=vec,
                payload=payload,
            )
        )

    for i in range(0, len(points), batch_size):
        client.upsert(collection_name=collection, points=points[i : i + batch_size])
