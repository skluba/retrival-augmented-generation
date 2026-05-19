"""Semantic cache cosine similarity and persistence (no Vertex calls)."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_pdf_app.rag.semantic_cache import (
    FileBackedSemanticCache,
    clear_semantic_cache_instances_for_tests,
    cosine_similarity,
)


def test_cosine_similarity_aligned_vectors() -> None:
    a = [1.0, 0.0, 0.0]
    b = [1.0, 0.0, 0.0]
    assert cosine_similarity(a, b) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal() -> None:
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_semantic_cache_roundtrip_and_threshold(tmp_path: Path) -> None:
    clear_semantic_cache_instances_for_tests()
    path = tmp_path / "sc.json"
    c = FileBackedSemanticCache(path)
    q = [1.0, 0.0, 0.0]
    c.put(q, "answer-one", max_entries=50)

    c2 = FileBackedSemanticCache(path)
    hit = c2.lookup_best([0.99, 0.01, 0.0], similarity_threshold=0.95)
    assert hit is not None
    assert hit[0] == "answer-one"
    assert hit[1] >= 0.95

    miss = c2.lookup_best([0.0, 1.0, 0.0], similarity_threshold=0.95)
    assert miss is None


def test_semantic_cache_respects_max_entries(tmp_path: Path) -> None:
    import json
    import time

    path = tmp_path / "sc2.json"
    c = FileBackedSemanticCache(path)
    for i in range(5):
        time.sleep(0.002)
        c.put([1.0, 0.0], f"a{i}", max_entries=3)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert len(data["entries"]) == 3
