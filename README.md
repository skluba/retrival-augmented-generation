[Quality gate](https://sonarcloud.io/summary/new_code?id=skluba_retrival-augmented-generation)

# Retrieval-Augmented Generation (PDF Lab)

Starter stack for a **Gemini-first PDF RAG** proof of concept:

- **LLM**: Gemini 2.0 Flash via **Vertex AI** and the official `**google-genai`** SDK (Application Default Credentials; no Gemini API keys).
- **API / UI**: **Streamlit** (`rag_pdf_app`).
- **Vector stores**: **FAISS** (bundled library, on-disk/index in `./data`), **Qdrant** (container).
- **RAG**: **LangChain** plus **Langfuse** (self-hosted traces) and **RAGAS** (evaluation tooling in code); **Phase 3** hybrid retrieval (BM25 + dense RRF) and **Phase 4** semantic answer cache plus LLM-guided multi-hop are **on by default** (opt out via `.env`).
- **PDFs**: **Docling**, **PyMuPDF**, multimodal Gemini flow to be layered on top.
- **Quality**: **SonarQube** (Compose) + CI with **pre-commit (ruff)**, **pytest**, and optional Sonar scanner.

---

## Docker Compose profiles & `COMPOSE_PROFILES`

Every service is gated by exactly one profile so you can tailor CPU/RAM use:


| Profile       | Services                                                                                                 |
| ------------- | -------------------------------------------------------------------------------------------------------- |
| `**core`**    | Streamlit `rag-app` + `**qdrant**`                                                                       |
| `**obs**`     | **Langfuse** (`langfuse-web`, `langfuse-worker`) + `**lf_*`** infra (Postgres, Redis, ClickHouse, MinIO) |
| `**quality**` | **Sonarqube** + its Postgres (`sonarqube`, `sonarqube_db`)                                               |


Docker Compose activates profiles from `**COMPOSE_PROFILES`** (comma-separated) in `.env`, or flags such as `--profile core`. The template sets `COMPOSE_PROFILES=core` so a plain `**docker compose up**` matches the slim daily loop.

Combine when needed:

```bash
docker compose --profile core --profile obs up --build -d
docker compose --profile core --profile quality up --build -d
docker compose --profile core --profile obs --profile quality up --build -d
```

`rag-app` declares **`USER app`** in the image (Sonar-friendly non-root default). **`docker-compose.yml` sets `user: "0:0"`** so the entrypoint can `chown` `/app/data` on the named volume, then **`runuser`** starts Streamlit as **`app` (UID/GID 1000)**. For plain `docker run` with the same volume pattern, pass **`--user 0:0`** once so that bootstrap runs; omit it only if `/app/data` is already writable by UID 1000.

Without `COMPOSE_PROFILES` **and** without `--profile`, no profiled containers start—which is deliberate.

---

## Prerequisites

1. Docker + Docker Compose plugin.
2. A **GCP project** with **Vertex AI API** enabled.
3. **ADC** credentials on the host (recommended for laptops):

```bash
gcloud auth application-default login
```

For unattended containers mount a **service account JSON** read-only into the stack and point `GOOGLE_APPLICATION_CREDENTIALS` at its in-container path (see Compose comments).

Copy env template:

```bash
cp .env.example .env   # populate GOOGLE_CLOUD_PROJECT and related values
```

1. **Locked dependencies** (CI and Docker use `uv sync --frozen`): after any change to `pyproject.toml`, refresh the lockfile with **Python 3.12** so versions stay reproducible:

```bash
uv lock --python 3.12
```

---

## Run the stack locally

After `cp .env.example .env`, ensure `**COMPOSE_PROFILES**` reflects what you need (defaults to `**core**` = app + Qdrant). Then:

```bash
docker compose up --build -d
```

Other examples appear in the profiles section above (`--profile obs`, `--profile quality`, or extend `COMPOSE_PROFILES` with comma-separated tokens).

Published host URLs (services must be enabled via profiles):


| Service                    | Purpose                    | Host URL / port                                                            |
| -------------------------- | -------------------------- | -------------------------------------------------------------------------- |
| `rag-app`                  | Streamlit UI               | [http://localhost:8501](http://localhost:8501)                             |
| `qdrant`                   | Dense vector DB            | REST [http://localhost:6333](http://localhost:6333) / gRPC `6334`          |
| `langfuse-web`             | Tracing dashboards         | [http://localhost:3000](http://localhost:3000)                             |
| MinIO console (`lf_minio`) | Langfuse S3 emulation      | [http://localhost:9091](http://localhost:9091) (API on `9090`)             |
| `sonarqube`                | Static analysis dashboards | [http://localhost:9010](http://localhost:9010) (`admin / admin` initially) |


> **Secrets**: Rotate every `postgres`, `redis`, `minio`, Langfuse crypto values before exposing anything beyond localhost. The Langfuse block started from upstream defaults with **host-rename adjustments** (`lf_*`) so Sonar Postgres can coexist.

> **Tracing without `obs`**: If Langfuse stays off, trace calls should stay disabled (`LANGFUSE_*` unset) so the app does not insist on resolving `langfuse-web` DNS.

Sonar project keys live in `sonar-project.properties` for local scanner / CI uploads.

Langfuse ingestion keys (`LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY`) appear after onboarding in Langfuse UI; drop them into `.env` for tracing from Python apps.

Dev install on the host (uses **uv** + `uv.lock`; [install uv](https://docs.astral.sh/uv/getting-started/installation/) if needed):

```bash
uv sync --frozen --extra dev --python 3.12
uv run pre-commit install
uv run pre-commit run --all-files   # mirrors CI style checks once
uv run pytest --maxfail=1 --disable-warnings
uv run streamlit run src/rag_pdf_app/streamlit_app.py
```

### Phase 2 · RAG evaluation (RAGAS + LLM judge)

After **Phase 1** ingestion built `FAISS_STORE_PATH` and populated Qdrant, evaluate retrieval + generation against the labeled workbook at the repository root: `RAG_evaluation_dataset-convertcsv.csv` (commit that file for shared runs). `**tests/fixtures/ifc_eval_sample.csv`** exercises the loader in CI when the full CSV is absent.

Reports include **`retrieval_config`** (hybrid on/off, pools, RRF weights, page filters, **FAISS directory basename only**) so you can diff **baseline vs Phase 3** runs without leaking host paths. See **`docs/eval/README.md`** for a recall-first tuning checklist (watch **context_recall** when tuning hybrid).

For strict labeled benchmarks where every row must retrieve fresh context (no similarity-cache shortcuts), set **`RAG_SEMANTIC_CACHE_ENABLED=false`** and typically **`RAG_MULTI_HOP_ENABLED=false`**, or pass **`rag-pdf-eval --disable-phase4`** once without editing `.env`.

Otherwise **`rag-pdf-eval`** uses the same **`Settings`** as Streamlit (Phase 3 hybrid + Phase 4 **on by default**; opt out in `.env` as above).

The CLI runs the same Phase 1 pipeline as Streamlit (dual retrieval → Gemini answer), scores outputs with **RAGAS** (faithfulness, answer relevancy, context precision vs reference answer, context recall), then applies an optional **Gemini judge** rubric on rows whose `Context_Content_Type` suggests tables, figures, or composite evidence.

```bash
# Full suite (Vertex ADC required; Qdrant reachable at QDRANT_URL)
uv run rag-pdf-eval

# Smoke subset + custom output path
uv run rag-pdf-eval --max-rows 3 --output-json ./reports/eval/smoke.json

# Strict eval: no semantic-cache hits, no multi-hop (single retrieval path per row)
uv run rag-pdf-eval --disable-phase4

# Skip judge; judge every row; Markdown only as JSON sidecar
uv run rag-pdf-eval --skip-judge
uv run rag-pdf-eval --judge-all
uv run rag-pdf-eval --no-output-markdown
```

Reports default to `reports/eval/phase2_rag_eval_<utc-timestamp>.json` with a **sibling `.md`** summary (retrieval settings, overall means, means by `Context_Content_Type`, judge stats, lowest-faithfulness rows). Pass `--no-output-markdown` if you only want JSON. The `reports/` directory is gitignored; copy the `.md` into `docs/eval/` when you want a rendered snapshot committed to the repo.

Run this **after each pipeline phase** you care about (e.g. after re-ingesting with new chunking or embedding settings) so regressions show up in the JSON deltas.

### Phase 3 · Hybrid retrieval & re-ranking (FAISS leg)

Phase 3 improves **FAISS** context selection (the **Qdrant** path stays dense-only for now). It is **on by default** (BM25 + dense fused via RRF). Set **`RAG_HYBRID_ENABLED=false`** for dense-only Phase 1 behaviour.

- **Sparse + dense hybrid**: an on-disk **BM25** index is built from the FAISS docstore and fused with dense neighbours via **reciprocal rank fusion (RRF)** with configurable leg weights (`rag_pdf_app/rag/sparse_bm25.py`, `hybrid_fusion.py`, `hybrid_retrieve.py`). For recall-first setups, increase **`RAG_RRF_SPARSE_WEIGHT`** slightly after widening **`RAG_HYBRID_*_POOL`** (see `docs/eval/README.md`).
- **Metadata**: chunks gain coarse **`content_type`** and **`section_hint`** during ingestion (`chunk_metadata.py`, `chunking.py`). Context headers in the UI include `content_type`.
- **Constraints**: optional **PDF page windows** come from env (`RAG_PAGE_FILTER_MIN` / `RAG_PAGE_FILTER_MAX`, 1-based inclusive) and/or inline queries such as `pages 10-20` (`query_page_window.py`, fused in `retrieve.py`).
- **Re-ranking**: after fusion, candidates get a light **metadata overlap boost**, then an optional **cross-encoder** pass if you install the **`phase3`** extra (`rerank_phase3.py`). There is no LLM-as-judge or graph re-ranker in-tree yet.

Hybrid retrieval is **on by default**. To use dense-only Phase 1 behaviour:

```bash
# In .env — see .env.example for the full list
RAG_HYBRID_ENABLED=false
```

Cross-encoder (pulls in PyTorch / `sentence-transformers`; omit if you only want BM25+dense RRF):

```bash
uv sync --frozen --extra dev --extra phase3 --python 3.12
# e.g. RAG_CROSS_ENCODER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
```

**Re-ingest** PDFs after upgrading Phase 3 chunk metadata if you want `content_type` / `section_hint` populated on every chunk in existing indexes.

### Phase 4 · Semantic cache & multi-hop retrieval

Phase 4 modules (`semantic_cache.py`, `multi_hop.py`) are **on by default**. Set **`RAG_SEMANTIC_CACHE_ENABLED=false`** and/or **`RAG_MULTI_HOP_ENABLED=false`** to disable them, or use **`rag-pdf-eval --disable-phase4`** for a single strict eval run.

- **Semantic cache**: embed the retrieval query (same string as dense retrieval after stripping inline page windows). On cosine similarity ≥ **`RAG_SEMANTIC_CACHE_SIMILARITY_THRESHOLD`** vs any cached embedding, return the stored answer and skip retrieval/generation. **Disabled** whenever inline `pages X–Y` clauses or **`RAG_PAGE_FILTER_*`** apply. **`RAG_SEMANTIC_CACHE_PATH`** is a template: if it ends with `.json`, actual files are `${stem}_${digest}.json` in the same directory for each **resolved FAISS directory + Qdrant collection** pair (no cross-corpus hits); otherwise it is a directory and files are `semantic_cache_${digest}.json` inside it (default template `./data/semantic_rag_cache.json`, under gitignored `./data`). **Shared hosts:** if multiple tenants reuse the same FAISS path and collection, or overwrite one index with different PDFs, cache entries can still mix—set **`RAG_SEMANTIC_CACHE_ENABLED=false`** or give each tenant isolated store paths / processes.
- **Multi-hop retrieval**: after the first dual retrieval, Gemini emits either **`DONE`** or one short follow-up query; a second **`retrieve_dual`** runs and **FAISS** hits are merged by `chunk_id` (Qdrant stays the first pass for side-by-side latency in the UI).

```bash
# In .env — see .env.example (defaults are on; uncomment to opt out)
# RAG_SEMANTIC_CACHE_ENABLED=false
# RAG_SEMANTIC_CACHE_SIMILARITY_THRESHOLD=0.92
# RAG_MULTI_HOP_ENABLED=false
```

---

## GitHub Actions / SonarQube

Workflow `.github/workflows/ci.yml`:

- **uv**: `uv sync --frozen --extra dev --python 3.12` so installs match the committed `**uv.lock`**.
- **pre-commit**: runs **ruff** lint + formatter via `.pre-commit-config.yaml` (`uv run pre-commit run --all-files`).
- **pytest**: `uv run pytest` plus `coverage.xml` and `reports/junit-report.xml` referenced by Sonar.
- **Sonar scanner**: runs on **push** to the repo **default branch** only (so `GITHUB_REF` resolves to `refs/heads/<default>`; pull_request runs use `refs/pull/...` and skip this step). It needs repository secrets `SONAR_TOKEN` and `**SONAR_HOST_URL`** reachable from GitHub-hosted runners (`**secrets.*` cannot be referenced in workflow `if:` expressions**, so the step is gated by branch/event only—the action still receives secrets via `env`).

Because GitHub-hosted runners cannot reach a Sonarqube container bound to `localhost` on your laptop, CI needs a URL that resolves on the internet (managed Sonarqube/SonarCloud ingress, VPN-hosted runner, or another reachable endpoint).

Repository secrets expected for uploads:


| Secret           | Meaning                                       |
| ---------------- | --------------------------------------------- |
| `SONAR_TOKEN`    | Project analysis token issued by SonarQube UI |
| `SONAR_HOST_URL` | External URL reachable from GitHub Runner     |


For pull requests originating from forks, scanners often cannot access private Sonar URLs—prefer **SonarCloud** or Runner inside your VPC.

---

## Feature branches

New features belong on isolated Git branches (see `.cursor/rules/feature-branch-workflow.mdc`). Keep PRs narrowly scoped around PDF ingestion, embeddings, retrieval, tracing, or eval layers.