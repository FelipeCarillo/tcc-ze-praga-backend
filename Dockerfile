FROM python:3.13-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

ENV UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=python3.13 \
    UV_CACHE_DIR=/tmp/uv-cache \
    UV_NO_SYNC=1 \
    HOME=/tmp

# Install dependencies first (cache layer)
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev

# Copy application code
COPY . .

RUN adduser --system --no-create-home appuser \
 && mkdir -p /tmp/uv-cache \
 && chown -R appuser /app/.venv /tmp/uv-cache
USER appuser

EXPOSE 8000

CMD ["sh", "-c", "\
  uv run alembic upgrade head && \
  uv run python -m scripts.seed_crops && \
  uv run python scripts/seed_action_plans.py && \
  uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 \
"]
