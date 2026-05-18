"""Application settings loaded from environment (never Gemini API keys; use GCP ADC)."""

from functools import lru_cache
from pathlib import Path

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
    vertex_embedding_model: str = Field(
        default="text-embedding-004",
        validation_alias=AliasChoices("VERTEX_EMBEDDING_MODEL"),
    )

    qdrant_url: str = "http://localhost:6333"
    faiss_store_path: str = "./data/faiss"
    rag_qdrant_collection: str = Field(
        default="ifc_annual_report_chunks",
        validation_alias=AliasChoices("RAG_QDRANT_COLLECTION"),
    )
    ifc_annual_report_pdf_path: str | None = Field(
        default=None,
        validation_alias=AliasChoices("IFC_ANNUAL_REPORT_PDF_PATH"),
    )
    rag_chunk_size: int = Field(
        default=1200,
        ge=200,
        validation_alias=AliasChoices("RAG_CHUNK_SIZE"),
    )
    rag_chunk_overlap: int = Field(
        default=200, ge=0, validation_alias=AliasChoices("RAG_CHUNK_OVERLAP")
    )
    rag_top_k: int = Field(default=5, ge=1, le=50, validation_alias=AliasChoices("RAG_TOP_K"))
    rag_embedding_batch_size: int = Field(
        default=250,
        ge=1,
        le=250,
        validation_alias=AliasChoices("RAG_EMBEDDING_BATCH_SIZE"),
        description="Vertex embedding predict limit is 250 texts per request.",
    )
    rag_embedding_max_input_tokens: int = Field(
        default=18_000,
        ge=1024,
        le=20_000,
        validation_alias=AliasChoices("RAG_EMBEDDING_MAX_INPUT_TOKENS"),
        description="Vertex caps total input tokens per embed request (sum over batch).",
    )

    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "http://localhost:3000"


@lru_cache
def get_settings() -> Settings:
    return Settings()


def clear_settings_cache() -> None:
    get_settings.cache_clear()


def bundled_ifc_annual_report_pdf_path() -> str | None:
    """Resolved path to the tracked IFC sample PDF at the repository root, if present."""

    root = Path(__file__).resolve().parents[2]
    candidate = root / "ifc-annual-report-2024-financials.pdf"
    return str(candidate) if candidate.is_file() else None
