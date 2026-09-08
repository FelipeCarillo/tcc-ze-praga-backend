"""Sobe a API para a demonstração local com o fluxo **real**.

Uso: ``uv run python -m scripts.serve_demo``

Diferença para os outros dois helpers:

- ``scripts.local_dev``  → Postgres local e **sem** LLM (valida contratos REST).
- ``scripts.serve_local`` → ``.env`` como está (exige Supabase no ar).
- ``scripts.serve_demo``  → Postgres local + **OpenAI real** + ONNX real +
  storage em disco. É o modo para apresentar o produto na própria máquina.

O que continua real aqui: classificação ONNX, agente LangGraph com streaming
SSE, HITL, planos de ação, memória semântica em pgvector e persistência.
O que muda: banco e imagens ficam na máquina, não no Supabase.

Só sobrescreve o que precisa apontar para local — ``OPENAI_API_KEY``,
``JWT_SECRET_KEY``, ``CHAT_MODEL`` e as flags de inferência continuam vindo do
``.env``. O segredo JWT precisa ser estável entre reinícios: é ele que assina
as URLs das imagens.
"""

from __future__ import annotations

import asyncio
import os
import selectors
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Aponta banco, storage e origens para a máquina local. Não toca em credenciais.
LOCAL_ENV = {
    "DATABASE_URL": "postgresql+asyncpg://postgres:postgres@127.0.0.1:5432/ze_praga",
    "STORAGE_BACKEND": "local",
    "LOCAL_STORAGE_DIR": str(REPO_ROOT / ".local-storage"),
    "PUBLIC_API_URL": "http://127.0.0.1:8000",
    "FRONTEND_URL": "http://127.0.0.1:3101",
    "ALLOWED_ORIGINS": (
        "http://127.0.0.1:3000,http://127.0.0.1:3100,http://127.0.0.1:3101,"
        "http://localhost:3000,http://localhost:3100,http://localhost:3101"
    ),
    # Sem Resend configurado, exigir verificação trancaria o login na hora ruim.
    "REQUIRE_EMAIL_VERIFICATION": "false",
    # Na demonstração todo acesso vem de 127.0.0.1 e cai no mesmo balde: com o
    # limite de produção (5 cadastros/hora) o apresentador se tranca sozinho ao
    # criar a segunda ou terceira conta. O freio continua ligado, só mais folgado.
    "RATE_LIMIT_LOGIN": "100/300",
    "RATE_LIMIT_REGISTER": "100/3600",
    "RATE_LIMIT_EMAIL": "50/3600",
    "RATE_LIMIT_RESET": "100/3600",
}


def selector_loop() -> asyncio.AbstractEventLoop:
    """psycopg async (checkpointer + Store pgvector) recusa o ProactorEventLoop.

    No Windows o default do asyncio é o Proactor, e o sintoma é traiçoeiro: a
    API sobe, ``/api/v1/health`` responde 200, as rotas REST funcionam — e só o
    chat, a retomada HITL e a memória semântica quebram.
    """
    return asyncio.SelectorEventLoop(selectors.SelectSelector())


def main() -> None:
    os.environ.update(LOCAL_ENV)

    # Importado só depois do update: Settings é instanciado no import de app.config.
    import uvicorn

    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))

    config = uvicorn.Config(
        "app.main:app",
        host=host,
        port=port,
        loop="none",  # não deixa o uvicorn instalar outro loop por cima
        log_level=os.environ.get("LOG_LEVEL", "info"),
    )
    server = uvicorn.Server(config)

    loop = selector_loop()
    asyncio.set_event_loop(loop)
    print(
        f"Zé Praga · demonstração local com fluxo real em http://{host}:{port}\n"
        f"  banco    : Postgres local (zepraga-ux-local)\n"
        f"  imagens  : {LOCAL_ENV['LOCAL_STORAGE_DIR']}\n"
        f"  ONNX     : real · agente: OpenAI real\n"
        f"  docs     : http://{host}:{port}/docs",
        flush=True,
    )
    try:
        loop.run_until_complete(server.serve())
    except KeyboardInterrupt:
        pass
    finally:
        loop.close()


if __name__ == "__main__":
    sys.exit(main())
