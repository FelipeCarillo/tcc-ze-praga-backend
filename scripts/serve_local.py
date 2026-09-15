"""Sobe a API local usando o ``.env`` real, no event loop que o Windows exige.

Uso: ``uv run python -m scripts.serve_local`` (porta 8000 por padrão).

Por que este módulo existe em vez de ``uv run uvicorn app.main:app``:
o checkpointer do LangGraph (``AsyncPostgresSaver``) e o Store do pgvector
(``AsyncPostgresStore``) falam psycopg async, e o psycopg **recusa** o
``ProactorEventLoop`` — que é o default do asyncio no Windows. Rodando pelo
uvicorn cru, o warm-up dos singletons falha no startup e a API sobe assim
mesmo: ``/api/v1/health`` responde 200 e as rotas REST funcionam, mas chat,
retomada HITL e memória semântica quebram no primeiro request. É uma falha
silenciosa justamente onde ela custa mais caro — no meio de uma demonstração.

Diferente de ``scripts.local_dev``, este helper **não** substitui variáveis:
usa o ``.env`` do repositório como está (Supabase, OpenAI, Storage reais).
Se a chave opcional do Resend estiver guardada em ``.env.local``, ela é
injetada isoladamente no processo da API. Nenhuma outra configuração desse
arquivo é aproveitada, evitando misturar configurações de demonstração com o
ambiente real.
"""

from __future__ import annotations

import asyncio
import os
import selectors
import sys
from pathlib import Path

from dotenv import dotenv_values


def selector_loop() -> asyncio.AbstractEventLoop:
    """Event loop compatível com psycopg async (ver docstring do módulo)."""
    return asyncio.SelectorEventLoop(selectors.SelectSelector())


def load_optional_local_resend_key() -> bool:
    """Disponibiliza somente a chave local do Resend, sem sobrescrever `.env`.

    A chave do ambiente é sempre prioritária. O fallback existe porque o
    ``.env.local`` é ignorado pelo Git e é o local adequado para um segredo
    opcional que não deve ser copiado para a configuração compartilhada.
    """
    if os.environ.get("RESEND_API_KEY"):
        return True

    local_env = Path(__file__).resolve().parents[1] / ".env.local"
    if not local_env.is_file():
        return False

    resend_api_key = dotenv_values(local_env).get("RESEND_API_KEY")
    if not resend_api_key:
        return False

    os.environ["RESEND_API_KEY"] = resend_api_key
    return True


def main() -> None:
    load_optional_local_resend_key()
    import uvicorn

    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))

    # `loop="none"` impede o uvicorn de instalar o loop dele por cima do nosso.
    config = uvicorn.Config(
        "app.main:app",
        host=host,
        port=port,
        loop="none",
        log_level=os.environ.get("LOG_LEVEL", "info"),
    )
    server = uvicorn.Server(config)

    loop = selector_loop()
    asyncio.set_event_loop(loop)
    print(
        f"Zé Praga · API local com .env real em http://{host}:{port} "
        f"· docs em http://{host}:{port}/docs",
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
