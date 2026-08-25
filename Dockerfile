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

# UID 1000 explícito: o Hugging Face Spaces roda o container com esse uid e
# precisa de permissão de escrita no cache. Um usuário "--system" ganharia uid
# < 1000 e quebraria lá, embora seguisse funcionando local — por isso o número
# fica fixo em vez de ficar a critério do adduser.
RUN adduser --uid 1000 --disabled-password --gecos "" appuser \
 && mkdir -p /tmp/uv-cache \
 && chown -R appuser /app/.venv /tmp/uv-cache
USER appuser

# Porta parametrizada: 8000 no docker-compose local, sobrescrita via PORT no
# Space quando necessário.
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "\
  uv run alembic upgrade head && \
  uv run python -m scripts.seed_crops && \
  uv run python scripts/seed_action_plans.py && \
  uv run uvicorn app.main:app --host 0.0.0.0 --port ${PORT} \
"]
