import pytest


@pytest.fixture(autouse=True)
def default_gcp_project(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project-qa")
