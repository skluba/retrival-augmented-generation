# Retrieval-Augmented Generation (PDF Lab)

Starter stack for a **Gemini-first PDF RAG** proof of concept:

- **LLM**: Gemini 2.0 Flash via **Vertex AI** and the official **`google-genai`** SDK (Application Default Credentials; no Gemini API keys).
- **API / UI**: **Streamlit** (`rag_pdf_app`).
- **Vector stores**: **FAISS** (bundled library, on-disk/index in `./data`), **Qdrant** (container).
- **RAG**: **LangChain** plus **Langfuse** (self-hosted traces) and **RAGAS** (evaluation tooling in code).
- **PDFs**: **Docling**, **PyMuPDF**, multimodal Gemini flow to be layered on top.
- **Quality**: **SonarQube** (Compose) + CI with **ruff/pytest**/Sonar scanner.

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

---

## Run the stack locally

Bring up infra + observability UI + Streamlit scaffold:

```bash
docker compose up --build rag-app qdrant sonarqube sonarqube_db langfuse-web
```

Recommended first pass (everything Compose defines):

```bash
docker compose up --build -d
```

Service map (published host ports):

| Service        | Purpose                     | Host URL / port                                      |
|----------------|-----------------------------|------------------------------------------------------|
| `rag-app`      | Streamlit UI                | <http://localhost:8501>                              |
| `qdrant`       | Dense vector DB             | REST <http://localhost:6333> / gRPC `6334`           |
| `langfuse-web` | Tracing dashboards          | <http://localhost:3000>                                |
| `minio-console`| Langfuse S3 emulation       | <http://localhost:9091> (API on `9090`)              |
| `sonarqube`    | Static analysis dashboards  | <http://localhost:9010> (`admin / admin` initially) |

> **Secrets**: Rotate every `postgres`, `redis`, `minio`, Langfuse crypto values before exposing anything beyond localhost. The Langfuse block started from upstream defaults with **host-rename adjustments** (`lf_*`) so Sonar Postgres can coexist.

Sonar project keys live in `sonar-project.properties` for local scanner / CI uploads.

Langfuse ingestion keys (`LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY`) appear after onboarding in Langfuse UI; drop them into `.env` for tracing from Python apps.

Dev install on the host:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
ruff format . && ruff check .
pytest --maxfail=1 --disable-warnings
streamlit run src/rag_pdf_app/streamlit_app.py
```

Gemini invocation uses **`google-genai`** with **`vertexai=True`** (wired in `rag_pdf_app/vertex_gemini.py`).

---

## GitHub Actions / SonarQube

Workflow `.github/workflows/ci.yml`:

- **Ruff**: lint (`ruff check`) + format enforcement (`ruff format --check`).
- **pytest**: executes the suite plus writes `coverage.xml` and `reports/junit-report.xml` referenced by Sonar.
- **Sonar scanner**: uploads only when:
  - the workflow runs against the repo **default branch** (for example after merges to `main`), and  
  - `SONAR_TOKEN` / `SONAR_HOST_URL` repository secrets exist.

Because GitHub-hosted runners cannot reach a Sonarqube container bound to `localhost` on your laptop, CI needs a URL that resolves on the internet (managed Sonarqube/SonarCloud ingress, VPN-hosted runner, or another reachable endpoint).

Repository secrets expected for uploads:

| Secret           | Meaning                                        |
|------------------|------------------------------------------------|
| `SONAR_TOKEN`    | Project analysis token issued by SonarQube UI  |
| `SONAR_HOST_URL` | External URL reachable from GitHub Runner      |

For pull requests originating from forks, scanners often cannot access private Sonar URLs—prefer **SonarCloud** or Runner inside your VPC.

---

## Feature branches

New features belong on isolated Git branches (see `.cursor/rules/feature-branch-workflow.mdc`). Keep PRs narrowly scoped around PDF ingestion, embeddings, retrieval, tracing, or eval layers.
