FROM python:3.12-slim-bookworm

# Runtime: ENTRYPOINT starts as root long enough to chown /app/data on volume mounts, then
# execs Streamlit (or CLI overrides) as UID/GID 1000 (`app`). Avoid leaving the service as root.

WORKDIR /app

# uv from Astral's distroless image (pinned semver tag = resolved release artefact), not `pip install`.
COPY --from=ghcr.io/astral-sh/uv:0.11.14 /uv /uvx /bin/

# Ghostscript supports Camelot (`parse_pdf_bytes(..., run_camelot=True)`). Parsing skips Camelot
# by default so production deployments that never enable it could omit this package.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ghostscript \
    libglib2.0-0 \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/* \
 && groupadd --gid 1000 app \
 && useradd --uid 1000 --gid app --home-dir /app --no-create-home --shell /usr/sbin/nologin app \
 && chown app:app /app

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

COPY --chown=app:app ifc-annual-report-2024-financials.pdf /app/
COPY --chown=app:app RAG_evaluation_dataset-convertcsv.csv /app/
COPY --chown=app:app pyproject.toml uv.lock README.md /app/
# `tool.uv.sources` points antlr4-python3-runtime + pylatexenc at vendored wheels so `--no-build`
# can satisfy the lockfile. The workspace root has no wheel — install it in a second step.
COPY --chown=app:app third_party/wheels /app/third_party/wheels
COPY --chown=app:app src /app/src

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

USER app

RUN uv sync --frozen --no-build --no-install-project --python 3.12 \
    && uv pip install --python 3.12 --no-deps .

USER root
# Strip write bits from baked-in corpus + source tree only (.venv stays writable for runtime caches).
RUN chmod a-w /app/ifc-annual-report-2024-financials.pdf \
        /app/RAG_evaluation_dataset-convertcsv.csv \
        /app/pyproject.toml \
        /app/uv.lock \
        /app/README.md \
    && chmod -R a-w /app/third_party/wheels /app/src

# Default USER stays root so ENTRYPOINT can chown volume-mounted /app/data (often root-owned from
# the host), then exec replaces PID 1 with the real command via `runuser -u app`. Streamlit and CLI
# overrides run as UID/GID 1000, not as root.

ENV PATH="/app/.venv/bin:$PATH" \
    VIRTUAL_ENV="/app/.venv" \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    PYTHONDONTWRITEBYTECODE=1

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]

EXPOSE 8501

CMD ["streamlit", "run", "src/rag_pdf_app/streamlit_app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
