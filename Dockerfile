FROM python:3.12-slim AS base

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Dependencies layer ──────────────────────────────────────────────────────
FROM base AS deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application layer ───────────────────────────────────────────────────────
FROM deps AS app
COPY . .
RUN pip install --no-cache-dir -e . --no-deps

# ── API image ───────────────────────────────────────────────────────────────
FROM app AS api
EXPOSE 8000
ENV HOST=0.0.0.0
ENV PORT=8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
CMD ["uvicorn", "llm_eval.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ── Dashboard image ─────────────────────────────────────────────────────────
FROM app AS dashboard
EXPOSE 8501
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1
CMD ["streamlit", "run", "llm_eval/dashboard/app.py", "--server.headless=true", "--server.port=8501", "--server.address=0.0.0.0"]
