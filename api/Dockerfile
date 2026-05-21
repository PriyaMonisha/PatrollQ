# filename: api/Dockerfile
# purpose:  PatrolIQ FastAPI inference server image

FROM python:3.11-slim

# Suppress .pyc files and enable stdout logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ARTIFACTS_DIR=/app/artifacts

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Install API dependencies only (not full requirements.txt)
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy API module and source modules needed for prediction + drift
COPY api/ ./api/
COPY src/ ./src/
COPY config.py .

# Non-root user for security
RUN useradd -m -u 1001 appuser
RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/v1/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
