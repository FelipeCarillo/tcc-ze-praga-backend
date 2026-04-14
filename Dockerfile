FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install dependencies first (cache layer)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Copy application code
COPY . .

EXPOSE 8000

CMD ["sh", "-c", "\
  uv run alembic upgrade head && \
  uv run python scripts/seed_action_plans.py && \
  uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 \
"]
