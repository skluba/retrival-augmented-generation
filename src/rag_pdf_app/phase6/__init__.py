"""Phase 6 — visual PDF patches + multilingual CLIP embeddings (ColPali-style scaffold).

Dense cosine retrieval indexes page patches into Qdrant; optional deterministic
embedding splits emulate late-interaction reranking via ColBERT-like MaxSim.
See README section \"Phase 6\" for deps and comparisons to Phase 1 (text-heavy RAG).

When ColPali-class models are plugged in behind the same ingest/retrieve API,
reuse :mod:`rag_pdf_app.phase6.late_interaction` helpers for MaxSim reranking scores.
"""

from __future__ import annotations
