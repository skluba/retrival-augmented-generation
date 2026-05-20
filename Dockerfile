FROM python:3.12-slim-bookworm

# Runtime: ENTRYPOINT starts as root long enough to chown /app/data on volume mounts, then
# execs Streamlit (or CLI overrides) as UID/GID 1000 (`app`). Avoid leaving the service as root.

WORKDIR /app

# uv from Astral's distroless image (pinned semver tag = resolved release artefact), not `pip install`.
COPY --from=ghcr.io/astral-sh/uv:0.11.14 /uv /uvx /bin/

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh

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
 && chmod 755 /usr/local/bin/docker-entrypoint.sh \
 && chown root:root /usr/local/bin/docker-entrypoint.sh

# Sonar/docker: prefer root-owned image layers with tight modes; install runs as root, then we drop privileges.
# Corpus/manifests: world-readable, root-writable until post-install chmod strips writes for non-.venv paths.
COPY --chown=root:root --chmod=644 \
    ifc-annual-report-2024-financials.pdf \
    RAG_evaluation_dataset-convertcsv.csv \
    pyproject.toml \
    uv.lock \
    README.md \
    /app/
# `tool.uv.sources` points antlr4-python3-runtime + pylatexenc at vendored wheels so `--no-build`
# can satisfy the lockfile. The workspace root has no wheel — install it in a second step.
# Trees need execute bits on directories for traversal — apply modes after COPY (see RUN below).
COPY --chown=root:root third_party/wheels /app/third_party/wheels
COPY --chown=root:root src /app/src

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

RUN uv sync --frozen --no-build --no-install-project --python 3.12 \
    && uv pip install --python 3.12 --no-deps . \
    && chmod 444 /app/ifc-annual-report-2024-financials.pdf \
        /app/RAG_evaluation_dataset-convertcsv.csv \
    && chmod 644 /app/pyproject.toml /app/uv.lock /app/README.md \
    && chmod -R a-w /app/third_party/wheels /app/src \
    && find /app/third_party/wheels -type d -exec chmod 555 {} + \
    && find /app/src -type d -exec chmod 555 {} + \
    && chown -R app:app /app/.venv \
    && chown app:app /app

ENV PATH="/app/.venv/bin:$PATH" \
    VIRTUAL_ENV="/app/.venv" \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    PYTHONDONTWRITEBYTECODE=1

# Sonar: non-root default USER. EntryPoint needs root once to `chown` volume-mounted `/app/data`;
# use Compose `user: "0:0"` for rag-app (see docker-compose.yml) or `docker run --user 0:0` with named volumes.
USER app

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]

EXPOSE 8501

CMD ["streamlit", "run", "src/rag_pdf_app/streamlit_app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
