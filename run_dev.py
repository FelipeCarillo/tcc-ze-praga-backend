"""Launcher de desenvolvimento — necessário no Windows.

O ``psycopg`` async, usado pelo checkpointer e pelo Store do LangGraph, não roda
no ``ProactorEventLoop``:

    psycopg.InterfaceError: Psycopg cannot use the 'ProactorEventLoop' [...]

E o uvicorn, no Windows, escolhe exatamente esse loop: ``asyncio_loop_factory``
devolve ``ProactorEventLoop`` sempre que não há subprocesso
(``uvicorn/loops/asyncio.py``). Como é uma *loop factory* e não uma *policy*,
chamar ``asyncio.set_event_loop_policy(...)`` antes não muda nada — nem dentro
de ``app/main.py``, nem antes de ``uvicorn.run()``.

Resultado sem este launcher: ``uv run uvicorn app.main:app`` sobe e responde
normalmente (o ``lifespan`` engole a exceção), mas o chat fica **sem
persistência de estado** — HITL não retoma — e a **memória semântica** em
pgvector nunca é escrita nem lida. Falha silenciosa.

Aqui montamos o loop na mão via ``asyncio.Runner(loop_factory=...)``, que tem
precedência sobre a escolha do uvicorn. O ``asyncpg`` (SQLAlchemy) funciona
igual nos dois loops.

Em Linux/Docker nada disso se aplica e ``uvicorn app.main:app`` basta.

Uso:
    uv run python run_dev.py [--host 127.0.0.1] [--port 8000] [--reload]
"""

from __future__ import annotations

import argparse
import asyncio
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Sobe a API em desenvolvimento.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Hot-reload. Delega ao uvicorn, que usa subprocesso e nesse caso "
        "já escolhe SelectorEventLoop sozinho.",
    )
    args = parser.parse_args()

    import uvicorn

    if args.reload:
        # Com --reload o uvicorn roda em subprocesso e o asyncio_loop_factory
        # devolve SelectorEventLoop — não precisa do Runner.
        uvicorn.run("app.main:app", host=args.host, port=args.port, reload=True)
        return

    config = uvicorn.Config("app.main:app", host=args.host, port=args.port)
    server = uvicorn.Server(config)

    if sys.platform == "win32":
        with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop) as runner:
            runner.run(server.serve())
    else:
        asyncio.run(server.serve())


if __name__ == "__main__":
    main()
