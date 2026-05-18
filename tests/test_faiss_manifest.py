"""FAISS manifest persistence avoids pickle-based load_local."""

import json

import pytest
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import FakeEmbeddings

from rag_pdf_app.config import Settings
from rag_pdf_app.rag.stores import load_faiss_index, save_faiss_index


def test_faiss_save_load_manifest_roundtrip(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project-qa")
    tmp = tmp_path_factory.mktemp("faiss_rt")
    settings = Settings(faiss_store_path=str(tmp / "idx"))

    fe = FakeEmbeddings(size=4)
    store = FAISS.from_texts(["alpha", "beta", "gamma"], fe)
    save_faiss_index(store, settings)

    loaded = load_faiss_index(fe, settings)
    assert loaded.index.ntotal == store.index.ntotal == 3
    first_id = loaded.index_to_docstore_id[0]
    doc = loaded.docstore.search(first_id)
    assert getattr(doc, "page_content", None) in {"alpha", "beta", "gamma"}

    manifest = json.loads((tmp / "idx" / "docstore.manifest.json").read_text(encoding="utf-8"))
    assert manifest["format"] == "rag_pdf_app_faiss_manifest_v1"
    assert (tmp / "idx" / "index.pkl").is_file() is False
