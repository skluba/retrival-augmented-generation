# RAG evaluation snapshots

Committed summaries for **Phase 2** (`rag-pdf-eval`) live here when you want Git-visible deltas. Raw JSON/Markdown artifacts default to `reports/eval/` (gitignored).

## Baseline vs Phase 3 hybrid (recall-focused)

1. **Same ingest**: one FAISS index + Qdrant collection; note commit SHA and PDF versions.
2. **Dense baseline**: `RAG_HYBRID_ENABLED=false`, fixed `RAG_TOP_K`, pools irrelevant.
3. **Hybrid run**: `RAG_HYBRID_ENABLED=true`; keep `RAG_TOP_K` identical for a fair comparison.

Each report now embeds a **`retrieval_config`** block (JSON and Markdown) plus per-row **`dual_retrieval_notes`** and **`rag_hybrid_enabled`**. Use that to prove runs were configured as intended. **`faiss_store_basename`** is the directory leaf name only so shared reports do not expose absolute paths; set **`RAG_EVAL_SNAPSHOT_INCLUDE_ABSOLUTE_PATHS=true`** only for private debugging.

Keep **`RAG_SEMANTIC_CACHE_ENABLED=false`** for standard labeled runs so similarity-cache hits do not shortcut retrieval during RAGAS.

### Tuning hybrid toward higher context recall

RAGAS **context_recall** is usually the first metric to react when retrieval misses gold passages.

Try, in order:

1. **Larger fusion pools** — `RAG_HYBRID_DENSE_POOL` and `RAG_HYBRID_SPARSE_POOL` (more candidates before RRF / re-rank trims to `RAG_TOP_K`).
2. **Lexical emphasis** — raise `RAG_RRF_SPARSE_WEIGHT` slightly (e.g. `1.1`–`1.25`) while leaving `RAG_RRF_DENSE_WEIGHT=1.0` so BM25 matches weigh more in RRF.
3. **RRF damping** — adjust `RAG_RRF_K` (typical range 40–90); smaller *k* tends to emphasize top ranks more aggressively.
4. **Defer precision tools** — leave `RAG_CROSS_ENCODER_MODEL` unset until recall is acceptable; cross-encoders can reorder away from breadth.

Copy the generated `.md` next to this README with a dated filename when publishing a snapshot.
