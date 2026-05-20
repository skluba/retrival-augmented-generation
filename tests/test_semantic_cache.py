"""Semantic cache cosine similarity and persistence (no Vertex calls)."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_pdf_app.rag.semantic_cache import (
    FileBackedSemanticCache,
    clear_semantic_cache_instances_for_tests,
    cosine_similarity,
    partitioned_semantic_cache_path,
)


def test_partitioned_semantic_cache_path_scopes_by_faiss_and_collection(tmp_path: Path) -> None:
    tpl = str(tmp_path / "semantic_rag_cache.json")
    a = tmp_path / "store_a"
    b = tmp_path / "store_b"
    a.mkdir()
    b.mkdir()
    p1 = partitioned_semantic_cache_path(
        cache_path_template=tpl,
        faiss_store_path=str(a),
        qdrant_collection="col_one",
    )
    p2 = partitioned_semantic_cache_path(
        cache_path_template=tpl,
        faiss_store_path=str(b),
        qdrant_collection="col_one",
    )
    p3 = partitioned_semantic_cache_path(
        cache_path_template=tpl,
        faiss_store_path=str(a),
        qdrant_collection="col_two",
    )
    assert p1 != p2
    assert p1 != p3
    assert p1.endswith(".json")
    assert Path(p1).parent == tmp_path


def test_partitioned_semantic_cache_path_directory_template(tmp_path: Path) -> None:
    d = tmp_path / "cache_root"
    d.mkdir()
    p = partitioned_semantic_cache_path(
        cache_path_template=str(d),
        faiss_store_path=str(tmp_path / "faiss"),
        qdrant_collection="c",
    )
    assert Path(p).parent == d
    assert "semantic_cache_" in Path(p).name


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
