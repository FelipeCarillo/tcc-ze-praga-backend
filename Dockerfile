FROM python:3.13-slim

# AWS Lambda Web Adapter: roda a mesma API FastAPI dentro do Lambda sem mudar
# código. É uma *extension* do Lambda — fora dele (Cloud Run, ECS,
# docker-compose) o arquivo fica inerte. Só funciona porque o chat é síncrono
# na nuvem: o modo BUFFERED do Lambda não faz SSE. Veja DEPLOY-ENXUTO.md.
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:1.1.0 /lambda-adapter /opt/extensions/lambda-adapter

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# UV_COMPILE_BYTECODE: gera os .pyc no build. Sem isso o primeiro import de
# langchain/langgraph compila tudo no cold start — e no Lambda, com o código
# em disco somente leitura, compilaria de novo a cada instância.
ENV UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=python3.13 \
    UV_CACHE_DIR=/tmp/uv-cache \
    UV_NO_SYNC=1 \
    UV_COMPILE_BYTECODE=1 \
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
 && chown -R appuser /app/.venv /tmp/uv-cache \
 && chmod +x scripts/entrypoint.sh
USER appuser

# Porta parametrizada: 8000 no docker-compose local; Cloud Run e Lambda Web
# Adapter leem a mesma variável PORT. O adapter só libera tráfego depois que o
# health responder — o warm-up do lifespan acontece antes do 1º usuário.
ENV PORT=8000 \
    RUN_MIGRATIONS_ON_BOOT=true \
    AWS_LWA_READINESS_CHECK_PATH=/api/v1/health
EXPOSE 8000

CMD ["sh", "scripts/entrypoint.sh"]
