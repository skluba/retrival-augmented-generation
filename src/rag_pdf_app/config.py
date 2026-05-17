"""Application settings loaded from environment (never Gemini API keys; use GCP ADC)."""

from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        validate_default=True,
    )

    google_cloud_project: str
    google_cloud_location: str = "us-central1"
    use_vertex_ai: bool = Field(
        default=True,
        validation_alias=AliasChoices("GOOGLE_GENAI_USE_VERTEXAI"),
    )
    vertex_generative_model: str = Field(
        default="gemini-2.0-flash",
        validation_alias=AliasChoices("VERTEX_GENERATIVE_MODEL"),
    )

    qdrant_url: str = "http://localhost:6333"
    faiss_store_path: str = "./data/faiss"

    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "http://localhost:3000"


@lru_cache
def get_settings() -> Settings:
    return Settings()


def clear_settings_cache() -> None:
    get_settings.cache_clear()
