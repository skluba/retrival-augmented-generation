"""Static UI copy for :mod:`streamlit_app` (Markdown, captions, labels, spinner text).

Keep user-controlled or corpus-derived strings out of Markdown paths in the lab layer;
those are routed through plaintext widgets elsewhere.
"""

from __future__ import annotations

import textwrap

# -----------------------------------------------------------------------------
# Shell (page / sidebar top level)
# -----------------------------------------------------------------------------

PAGE_TITLE = "PDF RAG Lab"
TITLE = "PDF RAG workspace"
CAPTION_MAIN = (
    "Gemini · Vertex AI · hybrid retrieval (FAISS + Qdrant) · tables · figure captions"
)
SIDEBAR_ENV_MARKDOWN_HEADING = "### Environment"
SIDEBAR_ADC_INSTRUCTIONS_TEXT = """\
Use GCP Application Default Credentials — no Gemini API keys.
Host: `gcloud auth application-default login`
Docker Compose: mount host ADC (see docker-compose `GCP_ADC_HOST_PATH`) \
and GOOGLE_APPLICATION_CREDENTIALS, or mount a service-account JSON."""
SIDEBAR_RELOAD_ENV_BUTTON = "Reload `.env` / env"
ERR_SETTINGS_LOAD = "Missing or invalid environment configuration."

# Tabs
TAB_LABELS = ("Overview", "Ingest & query (RAG)", "Phase 6 (visual patches)")

# -----------------------------------------------------------------------------
# Overview tab
# -----------------------------------------------------------------------------

OVERVIEW_SUCCESS = "Loaded settings from environment."
OVERVIEW_SMOKE_CHECKBOX_LABEL = "Smoke test Gemini (calls Vertex AI)"
OVERVIEW_PROMPT_LABEL = "Prompt"
OVERVIEW_PROMPT_DEFAULT = "Respond with exactly: pong"
OVERVIEW_GEMINI_RUN_BUTTON = "Run Gemini text"
OVERVIEW_GEMINI_SPINNER = "Calling Gemini…"
OVERVIEW_GEMINI_CREDENTIAL_ERROR = (
    "Google credentials are not available **inside this process**. "
    "On the host, run `gcloud auth application-default login`. "
    "In **Docker**, mount that JSON and set `GOOGLE_APPLICATION_CREDENTIALS` "
    "(see `docker-compose.yml` for `GCP_ADC_HOST_PATH`)."
)
OVERVIEW_FOOTNOTE_MARKDOWN = (
    "Upload a PDF, index **FAISS + Qdrant**, and ask questions in "
    "**Ingest & query (RAG)** — hybrid retrieval (dense + BM25), structured **table** "
    "chunks, **figure** caption chunks, optional semantic cache and multi-hop "
    "(see `.env`)."
)

# -----------------------------------------------------------------------------
# Phase 1 RAG tab
# -----------------------------------------------------------------------------

RAG_SUBHEADER = "Baseline RAG — text, tables (5.1), and figure captions (5.2)"
RAG_INTRO_MARKDOWN = textwrap.dedent(
    """
**Provide the PDF via upload** (normal flow). Ingest runs once (Vertex embeddings →
**FAISS** on disk + **Qdrant**). **Phase 5.1** adds **table chunks**; **Phase 5.2**
adds **figure / chart chunks** from Gemini image captions plus nearby PDF text.
By default (`RAG_IMAGE_REQUIRE_FIGURE_LABEL_NEARBY`, see `.env`) only rasters with a
nearby **`Figure N` / `Fig. N`** line are indexed—fewer bogus “images” beside plain
headings.
Each question retrieves on **both** backends; Gemini answers use **FAISS** hits.

**Phase 4 (`.env`, on by default):** `RAG_SEMANTIC_CACHE_ENABLED` reuses answers
for similar questions (skipped when using page-window filters).
`RAG_MULTI_HOP_ENABLED` runs a second retrieval pass after an LLM-suggested query.
Set either to `false` to disable.
"""
).strip()

RAG_UPLOAD_LABEL = "Upload PDF"
RAG_UPLOAD_HELP = "Preferred: choose the annual report (or any text PDF) from your machine."
RAG_LAYOUT_CHUNKS_CHECKBOX = "Structure-aware chunking (pdfminer)"
RAG_INGEST_BUTTON = "Ingest & index"
RAG_INGEST_SPINNER = "Embedding + FAISS + Qdrant…"
RAG_INGEST_UPLOAD_WARN = "Upload a PDF above to ingest."
RAG_INGEST_SUCCESS_TEMPLATE = (
    "Indexed **{chunk_count}** chunks · "
    "{embedding_dimensions}d · "
    "FAISS `{faiss_path}` · "
    "Qdrant `{qdrant_collection}`"
)
RAG_EXPANDER_INGEST_NOTES = "Ingest notes"
RAG_ERR_INGEST = "Ingest failed (Vertex, Qdrant reachability, or empty PDF)."

RAG_RELOAD_FAISS_BUTTON = "Reload FAISS from disk"
RAG_RELOAD_FAISS_SUCCESS = (
    "Loaded FAISS index — ensure Qdrant already has the same collection."
)
RAG_ERR_FAISS_LOAD = "Could not load FAISS store."

RAG_INFO_UPLOAD_OR_RELOAD_FAISS = (
    "Upload a PDF and run **Ingest & index**, or reload FAISS from disk to query."
)
RAG_QUESTION_LABEL = "Question about the report"
RAG_ASK_BUTTON = "Ask (retrieve + Gemini)"
RAG_QUERY_SPINNER = "Retrieving + generating…"
RAG_ERR_QUERY = "RAG query failed."

RAG_MARKDOWN_HEADING_ANSWER = "### Answer"
RAG_CHART_SUBHEADER = "Chart · retrieved table preview"
RAG_CAPTION_CHART_SKIP_NUMERIC = (
    "Numeric chart skipped — no numeric columns detected (table still shown above)."
)
RAG_CAPTION_NO_CSV_PREVIEW_TEMPLATE = (
    "Plotting: no CSV-backed table in top FAISS hits ({plot_note})."
)
RAG_SUCCESS_SEMANTIC_CACHE = (
    "Semantic cache hit — similar prior query (see retrieval notes)."
)
RAG_CAPTION_SEMANTIC_SIM_TEMPLATE = "Cache cosine similarity ≈ **{sim:.3f}**"

RAG_INFO_MULTI_HOP = "Multi-hop retrieval merged a second FAISS pass into context."
RAG_CAPTION_LANGFUSE_TEMPLATE = (
    "Langfuse trace: **{traced_label}** "
    "(needs `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY`)."
)
RAG_LANGFUSE_YES = "yes"
RAG_LANGFUSE_NO = "no"

RAG_METRIC_FAISS_LABEL = "FAISS retrieval ms"
RAG_METRIC_QDRANT_LABEL = "Qdrant retrieval ms"
RAG_FAISS_HIT_EXPANDER_TEMPLATE = (
    "FAISS · {kind} · {chunk_prefix} · score {score:.4f}"
)
RAG_QDRANT_HIT_EXPANDER_TEMPLATE = (
    "Qdrant · {chunk_prefix} · score {score:.4f}"
)
RAG_HIT_TEXT_PREVIEW_CHARS = 2000
RAG_EXPANDER_TRUNCATION_ELLIPSIS = "…"
RAG_CHUNK_PREFIX_LEN = 12

RAG_EXPANDER_FAISS_VS_QDRANT = "FAISS vs Qdrant — when to use which"
RAG_MARKDOWN_FAISS_VS_QDRANT = textwrap.dedent(
    """
**FAISS (local)** — Best for single-node prototypes, zero network hops after build,
simple files-on-disk deployment. Weaknesses: no multi-user ACLs, no incremental REST API,
you manage persistence and backups yourself.

**Qdrant (service)** — HTTP/gRPC API, horizontal scaling, filtering/payload queries,
multi-collection ops—better when several apps share vectors or you need managed ingestion.
Adds network latency vs purely local FAISS.

**This lab** keeps both in sync so you can compare raw query latency on identical embeddings;
scores differ because LangChain FAISS defaults to L2 distance while this Qdrant collection
uses cosine similarity.
"""
).strip()

RAG_EXPANDER_SCORE_INTERPRETATION = "Score interpretation"

DEFAULT_CHUNK_KIND = "narrative"


# -----------------------------------------------------------------------------
# Phase 6 — visual-patch tab (copy only; ingest/query logic stays in Streamlit layer)
# -----------------------------------------------------------------------------

PHASE6_ERR_MISSING_PILLOW = (
    "**Pillow** is missing (`pip install pillow` or "
    "`uv sync --frozen --extra dev --extra phase6`)."
)
PHASE6_ERR_MISSING_SENTENCE_TRANSFORMERS = (
    "**sentence-transformers** is missing — run "
    "`uv sync --frozen --extra dev --extra phase6 --python 3.12` (pulls Torch + CLIP)."
)
PHASE6_RELOAD_AFTER_EXTRA = " Reload Streamlit after install."
PHASE6_ENV_MARKDOWN_FALLBACK = (
    "Phase 6 uses `PHASE6_*` env vars — see [.env.example](.env.example) and README §Phase 6."
)


def phase6_optional_deps_hint() -> str | None:
    """Return install hint Markdown if Phase 6 optional deps are absent."""

    try:
        import PIL  # noqa: F401, PLC0415
    except ImportError:
        return PHASE6_ERR_MISSING_PILLOW

    try:
        import sentence_transformers  # noqa: F401, PLC0415
    except ImportError:
        return PHASE6_ERR_MISSING_SENTENCE_TRANSFORMERS
    return None


PHASE6_SUBHEADER = "Phase 6 — visual patches + multilingual CLIP (ColPali-style scaffold)"
PHASE6_INTRO_MARKDOWN = (
    "Raster PDF pages → **tiling crops** → **Sentence-Transformers CLIP** embeddings in a "
    "**separate Qdrant collection**. Retrieval uses cosine prefetch plus optional **pseudo "
    "MaxSim** (reshape pooled vectors into synthetic token rows). Gemini receives **PNG "
    "crops** rerendered from the PDF for transparency and layout fidelity vs Phase 5.2 "
    "figure **captions-only** indexing."
)

PHASE6_QDRANT_CAPTION_TEMPLATE = "Phase 6 Qdrant · `{collection}`"
PHASE6_UPLOAD_LABEL = "Upload PDF (Phase 6 · rebuilds Phase 6 collection)"
PHASE6_INGEST_BUTTON = "Ingest Phase 6 (CLIP patches)"
PHASE6_WARN_UPLOAD_FOR_INGEST = "Upload a PDF first."
PHASE6_INGEST_SPINNER = "Rasterising patches + CLIP + Qdrant…"
PHASE6_ERR_INGEST = "Phase 6 ingest failed (GPU/memory, Pillow, Torch, PDF)."
PHASE6_INGEST_SUCCESS_TEMPLATE = (
    "{patch_count} patches · "
    "{vector_dimension}d · "
    "Qdrant **`{collection}`** · "
    "`{embedding_model}`"
)

PHASE6_FORGET_SESSION_BUTTON = "Forget Phase 6 session PDF"
PHASE6_SESSION_CLEARED = "Phase 6 session PDF cleared."

PHASE6_CAPTION_SESSION_READY = (
    "Phase 6 session PDF loaded (basename + fingerprint below drive rerenders)."
)
PHASE6_SESSION_PLAINTEXT_TEMPLATE = (
    "basename: {basename}\nSHA256 prefix: {digest_disp} (thumbnails + Gemini crops)."
)

PHASE6_UPLOAD_DEFAULT_FILENAME = "upload.pdf"

PHASE6_QUESTION_LABEL = "Question (CLIP retrieval + Gemini over crops)"
PHASE6_SLIDER_RERENDER_DPI_LABEL = "Rerender DPI for Gemini crops / thumbnails"
PHASE6_SLIDER_RERENDER_DPI_HELP = (
    "Higher DPI improves serif text clarity; ingest raster DPI stays tied to PHASE6_RENDER_DPI."
)
PHASE6_ASK_BUTTON = "Retrieve + answer (Gemini multimodal)"
PHASE6_WARN_INGEST_FIRST = (
    "Ingest Phase 6 for this PDF first (or re-upload after reload)."
)
PHASE6_WARN_NO_DIGEST = "Missing PDF digest — run Phase 6 ingest."
PHASE6_WARN_EMPTY_QUESTION = "Enter a question."
PHASE6_SPINNER_CLIP_RETRIEVE = "CLIP retrieve + patch crops…"
PHASE6_EXPANDER_RETRIEVAL_TELEMETRY = "Retrieval telemetry"
PHASE6_WARN_NO_PATCHES = (
    "No patches returned — try another question, widen `PHASE6_VISUAL_PREFETCH`, ingest "
    "more pages, or lower `PHASE6_PATCH_STRIDE_PX` / `PHASE6_PATCH_SIZE_PX` overlap."
)
PHASE6_SPINNER_GEMINI = "Gemini multimodal answer…"
PHASE6_MARKDOWN_HEADING_ANSWER = "### Answer"
PHASE6_MARKDOWN_HEADING_SOURCES = "### Source attribution · rerendered crops"
PHASE6_ATTR_LINE_PREFIX_TEMPLATE = (
    "**[P{idx}]** · PDF page **`{page}`** · patch tile **r{r_ix}c{c_ix}**"
)
PHASE6_ATTR_SCORE_RERANK_TEMPLATE = (
    " · prefetch cos ≈ `{coarse:.4f}` · rerank `{score:.4f}`"
)
PHASE6_ATTR_SCORE_SIMPLE_TEMPLATE = " · score `{score:.4f}`"
PHASE6_ERR_QUERY = "Phase 6 query failed."

PHASE6_EXPANDER_COMPARE_PHASE1 = "Phase 1 vs Phase 6 — what to compare"
PHASE6_COMPARE_MARKDOWN = textwrap.dedent(
    """
**Phase 1 (+ 5.x)** indexes **tokens** — narrative/table JSON, optional **Gemini captions** for
figures (no vectors over pixels). Answers combine **hybrid textual** snippets only.

**Phase 6** indexes **CLIP patch embeddings** (+ optional pseudo-MaxSim rerank), then sends PNG
rerenders into Gemini — useful when OCR text is unreliable or visuals carry the semantics, at the
cost of heavier ingest (Torch) and heuristic layout tiling.

Pose the **same prompt** here and under **Ingest & query (RAG)**—if answers diverge, compare the
shown crops vs textual passages to diagnose lexical vs spatial evidence mismatch.
"""
).strip()


def rag_faiss_hit_chunk_prefix(chunk_id: str) -> str:
    cid = chunk_id[: RAG_CHUNK_PREFIX_LEN]
    return f"{cid}{RAG_EXPANDER_TRUNCATION_ELLIPSIS}"
