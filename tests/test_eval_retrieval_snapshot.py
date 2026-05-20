"""Eval retrieval_config must avoid leaking absolute filesystem paths by default."""

from __future__ import annotations

from pathlib import Path

from rag_pdf_app.config import Settings
from rag_pdf_app.eval.reporting import eval_retrieval_config_snapshot


def _minimal_settings(**kwargs: object) -> Settings:
    base: dict[str, object] = {
        "google_cloud_project": "test-proj",
        "faiss_store_path": "./data/faiss",
        # Do not inherit workspace .env toggles in CI / dev machines.
        "rag_eval_snapshot_include_absolute_paths": False,
    }
    base.update(kwargs)
    return Settings(**base)


def test_eval_retrieval_config_uses_faiss_basename_only() -> None:
    snap = eval_retrieval_config_snapshot(
        _minimal_settings(faiss_store_path="/Users/alice/myapp/custom_store/index_dir")
    )
    assert snap["faiss_store_basename"] == "index_dir"
    assert "faiss_store_path_absolute" not in snap
    assert "/Users/alice" not in str(snap.values())


def test_eval_retrieval_config_optional_absolute_path(tmp_path: Path) -> None:
    store = tmp_path / "nested" / "store"
    snap = eval_retrieval_config_snapshot(
        _minimal_settings(
            faiss_store_path=str(store),
            rag_eval_snapshot_include_absolute_paths=True,
        )
    )
    assert snap["faiss_store_basename"] == "store"
    assert snap["faiss_store_path_absolute"] == str(store.resolve())
