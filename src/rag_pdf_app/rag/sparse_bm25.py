"""BM25 sparse index built from the same chunks stored in a LangChain FAISS store."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


@dataclass(frozen=True)
class BM25ChunkIndex:
    """Chunk-aligned BM25 corpus (order matches ``chunk_ids``)."""

    chunk_ids: tuple[str, ...]
    _bm25: BM25Okapi

    @classmethod
    def from_texts(cls, chunk_ids: list[str], texts: list[str]) -> BM25ChunkIndex:
        if len(chunk_ids) != len(texts):
            raise ValueError("chunk_ids and texts length mismatch")
        corpus = [tokenize(t) for t in texts]
        return cls(chunk_ids=tuple(chunk_ids), _bm25=BM25Okapi(corpus))

    def top_chunk_scores(self, query: str, k: int) -> list[tuple[str, float]]:
        q = tokenize(query)
        scores = self._bm25.get_scores(q)
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
        out: list[tuple[str, float]] = []
        for idx, sc in ranked[:k]:
            out.append((self.chunk_ids[idx], float(sc)))
        return out


def _faiss_document_rows(store: FAISS) -> list[tuple[str, str, dict[str, Any]]]:
    docstore = store.docstore
    mapping = getattr(docstore, "_dict", None)
    if not isinstance(mapping, dict):
        return []
    rows: list[tuple[str, str, dict[str, Any]]] = []
    for doc in mapping.values():
        meta = dict(doc.metadata or {})
        cid = str(meta.get("chunk_id", "")).strip()
        if not cid:
            continue
        rows.append((cid, doc.page_content, meta))
    return rows


def build_bm25_index(store: FAISS) -> BM25ChunkIndex | None:
    rows = _faiss_document_rows(store)
    if not rows:
        return None
    ids = [r[0] for r in rows]
    texts = [r[1] for r in rows]
    return BM25ChunkIndex.from_texts(ids, texts)


def find_chunk_by_id(store: FAISS, chunk_id: str) -> tuple[str, dict[str, Any]] | None:
    for cid, text, meta in _faiss_document_rows(store):
        if cid == chunk_id:
            return text, meta
    return None


def faiss_snapshot_cache_key(faiss_root: str | Path) -> str:
    root = Path(faiss_root).resolve()
    newest = 0.0
    if root.is_dir():
        for child in root.iterdir():
            if child.is_file():
                newest = max(newest, child.stat().st_mtime)
    return f"{root}:{newest:.9f}"


_BM25_CACHE: dict[str, BM25ChunkIndex] = {}


def get_cached_bm25_index(store: FAISS, cache_key: str) -> tuple[BM25ChunkIndex | None, float, bool]:
    """Return (index, build_seconds, cache_hit)."""

    if cache_key in _BM25_CACHE:
        return _BM25_CACHE[cache_key], 0.0, True
    t0 = time.perf_counter()
    idx = build_bm25_index(store)
    dt = time.perf_counter() - t0
    if idx is not None:
        _BM25_CACHE[cache_key] = idx
    return idx, dt, False


def clear_bm25_cache() -> None:
    _BM25_CACHE.clear()
