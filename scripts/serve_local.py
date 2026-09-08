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
"""

from __future__ import annotations

import asyncio
import os
import selectors
import sys


def selector_loop() -> asyncio.AbstractEventLoop:
    """Event loop compatível com psycopg async (ver docstring do módulo)."""
    return asyncio.SelectorEventLoop(selectors.SelectSelector())


def main() -> None:
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
