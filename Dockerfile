FROM python:3.12-slim-bookworm

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
 && rm -rf /var/lib/apt/lists/*

COPY ifc-annual-report-2024-financials.pdf /app/
COPY pyproject.toml uv.lock README.md /app/
# `tool.uv.sources` points antlr4-python3-runtime + pylatexenc at vendored wheels so `--no-build`
# can satisfy the lockfile. The workspace root has no wheel — install it in a second step.
COPY third_party/wheels /app/third_party/wheels
COPY src /app/src

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

RUN uv sync --frozen --no-build --no-install-project --python 3.12 \
    && uv pip install --python 3.12 --no-deps .

ENV PATH="/app/.venv/bin:$PATH" \
    VIRTUAL_ENV="/app/.venv" \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8501

CMD ["streamlit", "run", "src/rag_pdf_app/streamlit_app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
