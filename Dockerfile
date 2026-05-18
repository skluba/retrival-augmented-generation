FROM python:3.12-slim-bookworm

WORKDIR /app

# Ghostscript supports Camelot (`parse_pdf_bytes(..., run_camelot=True)`). Parsing skips Camelot
# by default so production deployments that never enable it could omit this package.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ghostscript \
    libglib2.0-0 \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "uv==0.11.14"

COPY pyproject.toml uv.lock README.md /app/
COPY ifc-annual-report-2024-financials.pdf /app/
COPY src /app/src

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

RUN uv sync --frozen --python 3.12

ENV PATH="/app/.venv/bin:$PATH" \
    VIRTUAL_ENV="/app/.venv" \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8501

CMD ["streamlit", "run", "src/rag_pdf_app/streamlit_app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
