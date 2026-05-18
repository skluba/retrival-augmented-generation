from rag_pdf_app.config import Settings, clear_settings_cache, get_settings


def test_settings_roundtrip_via_cache() -> None:
    clear_settings_cache()
    first = get_settings()
    clear_settings_cache()
    second = get_settings()

    assert first.google_cloud_project == second.google_cloud_project == "test-project-qa"
    assert first.use_vertex_ai is True


def test_settings_direct_instantiation() -> None:
    settings = Settings()
    assert settings.vertex_generative_model
    assert settings.vertex_embedding_model
    assert settings.qdrant_url.startswith("http")
