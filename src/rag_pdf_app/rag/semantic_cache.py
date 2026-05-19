"""File-backed semantic cache for RAG answers (Phase 4).

Caches (query embedding → answer) pairs. Lookup uses cosine similarity against stored
embeddings. Disabled when page-window constraints apply (see ``query.run_phase1_rag``).

Does **not** persist raw query strings by default (embedding + answer only).
"""

from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path
from typing import Any

_SCHEMA_VERSION = 1


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class FileBackedSemanticCache:
    """JSON file cache; thread-safe load/save with simple LRU eviction by ``created_ts``."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._entries: list[dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if not self._path.is_file():
            self._entries = []
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self._entries = []
            return
        if raw.get("version") != _SCHEMA_VERSION:
            self._entries = []
            return
        entries = raw.get("entries")
        self._entries = entries if isinstance(entries, list) else []

    def _save_unlocked(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": _SCHEMA_VERSION, "entries": self._entries}
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self._path)

    def lookup_best(
        self,
        query_embedding: list[float],
        *,
        similarity_threshold: float,
    ) -> tuple[str, float] | None:
        """Return ``(answer, similarity)`` for the best entry at or above threshold."""

        best_sim = -1.0
        best_answer: str | None = None
        with self._lock:
            for row in self._entries:
                emb = row.get("embedding")
                ans = row.get("answer")
                if not isinstance(emb, list) or not emb or not isinstance(ans, str):
                    continue
                try:
                    sim = cosine_similarity(query_embedding, [float(x) for x in emb])
                except (TypeError, ValueError):
                    continue
                if sim > best_sim:
                    best_sim = sim
                    best_answer = ans
        if best_answer is None or best_sim < similarity_threshold:
            return None
        return best_answer, best_sim

    def put(
        self,
        query_embedding: list[float],
        answer: str,
        *,
        max_entries: int,
    ) -> None:
        """Append an entry; evict oldest rows when over ``max_entries``."""

        row = {
            "embedding": list(query_embedding),
            "answer": answer,
            "created_ts": time.time(),
        }
        with self._lock:
            self._entries.append(row)
            if len(self._entries) > max_entries:
                self._entries.sort(key=lambda r: float(r.get("created_ts", 0.0)))
                overflow = len(self._entries) - max_entries
                if overflow > 0:
                    self._entries = self._entries[overflow:]
            self._save_unlocked()


_CACHE_LOCK = threading.Lock()
_CACHES: dict[str, FileBackedSemanticCache] = {}


def get_semantic_cache(path_str: str) -> FileBackedSemanticCache:
    """Singleton per resolved path (process-local)."""

    key = str(Path(path_str).expanduser().resolve())
    with _CACHE_LOCK:
        if key not in _CACHES:
            _CACHES[key] = FileBackedSemanticCache(Path(key))
        return _CACHES[key]


def clear_semantic_cache_instances_for_tests() -> None:
    """Drop process-local cache handles (pytest isolation)."""

    with _CACHE_LOCK:
        _CACHES.clear()
