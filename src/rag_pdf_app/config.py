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
        populate_by_name=True,
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

    rag_hybrid_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("RAG_HYBRID_ENABLED"),
    )
    rag_hybrid_dense_pool: int = Field(
        default=24,
        ge=5,
        le=200,
        validation_alias=AliasChoices("RAG_HYBRID_DENSE_POOL"),
    )
    rag_hybrid_sparse_pool: int = Field(
        default=24,
        ge=5,
        le=200,
        validation_alias=AliasChoices("RAG_HYBRID_SPARSE_POOL"),
    )
    rag_rrf_k: int = Field(default=60, ge=1, le=300, validation_alias=AliasChoices("RAG_RRF_K"))
    rag_rrf_dense_weight: float = Field(
        default=1.0,
        ge=0.1,
        le=5.0,
        validation_alias=AliasChoices("RAG_RRF_DENSE_WEIGHT"),
        description="RRF weight for the dense (FAISS) ranking leg.",
    )
    rag_rrf_sparse_weight: float = Field(
        default=1.0,
        ge=0.1,
        le=5.0,
        validation_alias=AliasChoices("RAG_RRF_SPARSE_WEIGHT"),
        description=(
            "RRF weight for the sparse (BM25) leg; raise slightly (e.g. 1.15) for recall."
        ),
    )
    rag_cross_encoder_model: str | None = Field(
        default=None,
        validation_alias=AliasChoices("RAG_CROSS_ENCODER_MODEL"),
    )
    rag_cross_encoder_top_n: int = Field(
        default=24,
        ge=1,
        le=120,
        validation_alias=AliasChoices("RAG_CROSS_ENCODER_TOP_N"),
    )
    rag_metadata_boost_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("RAG_METADATA_BOOST_ENABLED"),
    )
    rag_page_filter_min: int | None = Field(
        default=None,
        ge=1,
        validation_alias=AliasChoices("RAG_PAGE_FILTER_MIN"),
        description="Optional 1-based inclusive PDF page lower bound for retrieval filtering.",
    )
    rag_page_filter_max: int | None = Field(
        default=None,
        ge=1,
        validation_alias=AliasChoices("RAG_PAGE_FILTER_MAX"),
        description="Optional 1-based inclusive PDF page upper bound for retrieval filtering.",
    )
    rag_eval_snapshot_include_absolute_paths: bool = Field(
        default=False,
        validation_alias=AliasChoices("RAG_EVAL_SNAPSHOT_INCLUDE_ABSOLUTE_PATHS"),
        description=(
            "If true, embed resolved absolute FAISS directory in eval retrieval_config "
            "(avoid when sharing reports)."
        ),
    )

    # Phase 4 · Semantic answer cache (embedding similarity; disabled with page-window filters).
    rag_semantic_cache_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("RAG_SEMANTIC_CACHE_ENABLED"),
        description=(
            "Answer cache keyed by query embedding; files are partitioned by resolved FAISS store "
            "path and Qdrant collection. On shared Streamlit workers, reuse of the same paths "
            "across different uploads still shares one cache—disable for strict multi-tenant "
            "hosts or isolate indices per tenant."
        ),
    )
    rag_semantic_cache_path: str = Field(
        default="./data/semantic_rag_cache.json",
        validation_alias=AliasChoices("RAG_SEMANTIC_CACHE_PATH"),
    )
    rag_semantic_cache_similarity_threshold: float = Field(
        default=0.92,
        ge=0.5,
        le=1.0,
        validation_alias=AliasChoices("RAG_SEMANTIC_CACHE_SIMILARITY_THRESHOLD"),
        description="Cosine similarity minimum vs cached query embeddings to reuse an answer.",
    )
    rag_semantic_cache_max_entries: int = Field(
        default=256,
        ge=16,
        le=10_000,
        validation_alias=AliasChoices("RAG_SEMANTIC_CACHE_MAX_ENTRIES"),
    )

    rag_multi_hop_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("RAG_MULTI_HOP_ENABLED"),
        description="Second retrieval pass using an LLM-suggested follow-up query (FAISS merge).",
    )

    # Phase 5.1 · Table indexing (parse PDF into table chunks alongside narrative windows).
    rag_table_indexing_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("RAG_TABLE_INDEXING_ENABLED"),
        description=(
            "During ingest, run structured PDF parsing to emit extra chunks for detected tables "
            "(PyMuPDF; optional Camelot when enabled)."
        ),
    )
    rag_ingest_run_camelot: bool = Field(
        default=False,
        validation_alias=AliasChoices("RAG_INGEST_RUN_CAMELOT"),
        description="Camelot/Ghostscript table extraction for trusted PDFs only.",
    )
    rag_ingest_run_docling: bool = Field(
        default=True,
        validation_alias=AliasChoices("RAG_INGEST_RUN_DOCLING"),
        description="Optional Docling markdown span during ingest (heavy; improves some layouts).",
    )
    rag_ingest_gemini_table_summaries: bool = Field(
        default=True,
        validation_alias=AliasChoices("RAG_INGEST_GEMINI_TABLE_SUMMARIES"),
        description="Gemini short summaries for noisy tables before embedding.",
    )
    rag_plotting_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("RAG_PLOTTING_ENABLED"),
        description="Streamlit: when retrieved passages include CSV previews, offer simple charts.",
    )

    # Phase 5.2 · Figure / chart chunks (captions + nearby text indexed for retrieval).
    rag_image_indexing_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("RAG_IMAGE_INDEXING_ENABLED"),
        description=(
            "During ingest, index Gemini (or fallback) descriptions of raster figures plus "
            "neighbouring text as searchable chunks."
        ),
    )
    rag_ingest_gemini_image_captions: bool = Field(
        default=True,
        validation_alias=AliasChoices("RAG_INGEST_GEMINI_IMAGE_CAPTIONS"),
        description="Vertex Gemini captions for extracted images during ingest.",
    )
    rag_image_index_min_area_px: int = Field(
        default=8192,
        ge=0,
        le=4_000_000,
        validation_alias=AliasChoices("RAG_IMAGE_INDEX_MIN_AREA_PX"),
        description=(
            "Skip raster images smaller than width×height area unless they carry caption/text cues."
        ),
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


def clear_settings_cache() -> None:
    get_settings.cache_clear()
