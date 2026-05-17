FROM python:3.12-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libglib2.0-0 \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md /app/
COPY src /app/src

RUN pip install --no-cache-dir -e .

ENV PYTHONUNBUFFERED=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8501

CMD ["streamlit", "run", "src/rag_pdf_app/streamlit_app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
