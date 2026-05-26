"""Checkpointer singleton (TCC-058 / Sprint A4.5).

Wrapper assincrono em torno de ``langgraph.checkpoint.postgres.aio.AsyncPostgresSaver``
pra persistir snapshots de execucao do grafo — pre-requisito do ciclo HITL
``ask_user`` + ``Command(resume=...)``.

Distincao com o Store (``app.db.store``):
- ``Store`` persiste **memorias** indexadas por namespace
  (ex: ``("user", uid, "diagnoses")``) — escopo cross-thread.
- ``Checkpointer`` persiste o **estado** do grafo por ``thread_id`` —
  escopo dentro de uma sessao, necessario pra retomar interrupts.

Lifecycle: ``get_checkpointer()`` retorna um singleton lazy-initialized.
``setup()`` (cria tabelas/indices no Postgres) eh idempotente e roda na
primeira chamada. ``close_checkpointer()`` deve ser chamado no shutdown
do FastAPI lifespan.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, cast

from app.config import settings

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver


# ── Singleton holder ──────────────────────────────────────────────────────────

_holder: dict[str, object] = {
    "checkpointer": None,
    "cm": None,
    "lock": None,
}


def _make_conn_string() -> str:
    """Converte a DSN async (``postgresql+asyncpg://...``) pra psycopg.

    O ``AsyncPostgresSaver`` usa psycopg async internamente (precisa de
    ``postgresql://`` sem o ``+asyncpg``).
    """
    url = settings.database_url
    if url.startswith("postgresql+asyncpg://"):
        url = url.replace("postgresql+asyncpg://", "postgresql://", 1)
    elif url.startswith("postgres+asyncpg://"):
        url = url.replace("postgres+asyncpg://", "postgresql://", 1)
    return url


@asynccontextmanager
async def open_checkpointer() -> AsyncIterator[AsyncPostgresSaver]:
    """Context manager pra abrir um saver novo (uso em testes e scripts).

    Em runtime, prefira ``get_checkpointer()`` que cacheia singleton.
    """
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    conn_string = _make_conn_string()
    async with AsyncPostgresSaver.from_conn_string(conn_string) as saver:
        await saver.setup()
        yield saver


async def get_checkpointer() -> AsyncPostgresSaver:
    """Retorna o ``AsyncPostgresSaver`` singleton — inicializa na primeira chamada.

    Thread-safe via ``asyncio.Lock``. Em testes, monkeypatch este helper pra
    injetar ``MemorySaver``.
    """
    if _holder["lock"] is None:
        _holder["lock"] = asyncio.Lock()

    lock = _holder["lock"]
    assert isinstance(lock, asyncio.Lock)

    async with lock:
        if _holder["checkpointer"] is None:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

            cm = AsyncPostgresSaver.from_conn_string(_make_conn_string())
            saver = await cm.__aenter__()
            await saver.setup()
            _holder["checkpointer"] = saver
            _holder["cm"] = cm

    active = _holder["checkpointer"]
    assert active is not None
    return cast("AsyncPostgresSaver", active)


async def close_checkpointer() -> None:
    """Fecha o saver singleton — chamar no lifespan shutdown do FastAPI."""
    cm = _holder.get("cm")
    if cm is not None:
        await cm.__aexit__(None, None, None)  # type: ignore[attr-defined]
    _holder["checkpointer"] = None
    _holder["cm"] = None


def reset_checkpointer_for_tests() -> None:
    """Reseta o singleton — uso EXCLUSIVO em testes."""
    _holder["checkpointer"] = None
    _holder["cm"] = None
    _holder["lock"] = None
