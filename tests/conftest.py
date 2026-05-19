import os

# macOS: PyTorch (pulled by optional RAG deps) and faiss both link libomp; without this,
# combined imports abort during FAISS search (OMP: duplicate libomp).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pytest


@pytest.fixture(autouse=True)
def default_gcp_project(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project-qa")
