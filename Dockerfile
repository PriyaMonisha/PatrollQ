# filename: Dockerfile
# purpose:  PatrolIQ Streamlit dashboard — single-stage image
# version:  1.0
#
# Build:  docker build -t patroliq .
# Run:    docker run -p 8501:8501 patroliq
# Compose: docker compose up -d

FROM python:3.11-slim
WORKDIR /app

ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1001 appuser

COPY streamlit-requirements.txt .
RUN pip install --no-cache-dir -r streamlit-requirements.txt

COPY streamlit_app.py .
COPY pages/ pages/
COPY config.py .
COPY src/__init__.py src/
COPY src/utils/ src/utils/
COPY artifacts/ artifacts/

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "streamlit_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
