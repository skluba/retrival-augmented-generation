"""Phase 1 baseline RAG: chunk → embed → FAISS + Qdrant → retrieve → Gemini."""

from rag_pdf_app.rag.ingest import IngestOutcome, ingest_pdf_bytes_to_indexes
from rag_pdf_app.rag.models import TextChunk
from rag_pdf_app.rag.query import Phase1RagResult, run_phase1_rag

__all__ = [
    "IngestOutcome",
    "Phase1RagResult",
    "TextChunk",
    "ingest_pdf_bytes_to_indexes",
    "run_phase1_rag",
]
