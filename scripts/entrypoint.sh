#!/bin/sh
# Entrypoint da imagem do backend.
#
#   (sem argumento)  sobe a API. Com RUN_MIGRATIONS_ON_BOOT=true (padrão, o que
#                    o docker-compose local espera) aplica migrations e seeds
#                    antes. Na nuvem deixe false: cada cold start pagaria
#                    alembic + seeds, e o Lambda ainda os repetiria por instância.
#   migrate          só aplica migrations e seeds e sai — rode uma vez por
#                    deploy que trouxer migration nova:
#                    docker run --rm --env-file .env.cloud <imagem> migrate
#
# Chama os binários da .venv direto em vez de `uv run`: o uv tentaria escrever
# cache/lock em disco, e o Lambda só deixa escrever em /tmp.
set -e

VENV=/app/.venv/bin

migrar() {
  "$VENV/alembic" upgrade head
  "$VENV/python" -m scripts.seed_crops
  "$VENV/python" scripts/seed_action_plans.py
}

if [ "$1" = "migrate" ]; then
  migrar
  exit 0
fi

if [ "${RUN_MIGRATIONS_ON_BOOT:-true}" = "true" ]; then
  migrar
fi

exec "$VENV/uvicorn" app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
