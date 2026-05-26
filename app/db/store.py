"""Long-term memory store (TCC-044 / Sprint A2.5).

Wrapper assincrono em torno de ``langgraph.store.postgres.AsyncPostgresStore``
pra prover memoria semantica cross-session: diagnoses passados, resumos de
sessao, preferencias do usuario, etc.

O Store eh distinto do Checkpointer: checkpointer persiste o **estado** do
grafo num thread_id (sessao); store persiste **memorias** indexadas por
namespace (ex: ``("user", uid, "diagnoses")``).

Indexacao: embeddings sao gerados pelo ``OpenAIEmbeddings`` configurado
em ``settings.openai_embeddings_model`` (default ``text-embedding-3-small``,
1536 dims).

Lifecycle: a coroutine ``get_store()`` retorna um singleton lazy-initialized.
O ``setup()`` (que cria tabelas/indices no Postgres) eh idempotente e roda
na primeira chamada.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, cast

from app.config import settings

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from langgraph.store.postgres.aio import AsyncPostgresStore
    from langgraph.store.postgres.base import PostgresIndexConfig


# ── Singleton holder ──────────────────────────────────────────────────────────

# Holds the active AsyncPostgresStore + the cm context manager keeping its
# connection pool alive. Reset on lifespan shutdown via ``close_store()``.
_store_holder: dict[str, object] = {"store": None, "cm": None, "lock": None}


def _make_conn_string() -> str:
    """Converte a DSN async (``postgresql+asyncpg://...``) pra psycopg sync.

    O ``AsyncPostgresStore`` usa psycopg async internamente (precisa de
    ``postgresql://`` sem o ``+asyncpg``).
    """
    url = settings.database_url
    if url.startswith("postgresql+asyncpg://"):
        url = url.replace("postgresql+asyncpg://", "postgresql://", 1)
    elif url.startswith("postgres+asyncpg://"):
        url = url.replace("postgres+asyncpg://", "postgresql://", 1)
    return url


def _build_index_config() -> PostgresIndexConfig:
    """Constroi o ``PostgresIndexConfig`` pra vector search."""
    from langchain_openai import OpenAIEmbeddings
    from pydantic import SecretStr

    embeddings = OpenAIEmbeddings(
        model=settings.openai_embeddings_model,
        api_key=SecretStr(settings.openai_api_key) if settings.openai_api_key else None,
    )
    return {
        "dims": settings.openai_embeddings_dims,
        "embed": embeddings,
        # Indexa todos os campos "summary_text" automaticamente — outros
        # campos do value sao armazenados sem embedding.
        "fields": ["summary_text"],
    }


@asynccontextmanager
async def open_store() -> AsyncIterator[AsyncPostgresStore]:
    """Context manager pra abrir um Store novo (uso em testes e scripts).

    Em runtime, prefira ``get_store()`` que cacheia singleton.
    """
    from langgraph.store.postgres.aio import AsyncPostgresStore

    conn_string = _make_conn_string()
    async with AsyncPostgresStore.from_conn_string(
        conn_string,
        index=_build_index_config(),
    ) as store:
        await store.setup()
        yield store


async def get_store() -> AsyncPostgresStore:
    """Retorna o ``AsyncPostgresStore`` singleton — inicializa na primeira chamada.

    Thread-safe via ``asyncio.Lock``. Em testes, monkeypatch este helper pra
    injetar mock store.
    """
    if _store_holder["lock"] is None:
        _store_holder["lock"] = asyncio.Lock()

    lock = _store_holder["lock"]
    assert isinstance(lock, asyncio.Lock)

    async with lock:
        if _store_holder["store"] is None:
            from langgraph.store.postgres.aio import AsyncPostgresStore

            cm = AsyncPostgresStore.from_conn_string(
                _make_conn_string(),
                index=_build_index_config(),
            )
            store = await cm.__aenter__()
            await store.setup()
            _store_holder["store"] = store
            _store_holder["cm"] = cm

    active = _store_holder["store"]
    assert active is not None
    return cast("AsyncPostgresStore", active)


async def close_store() -> None:
    """Fecha o Store singleton — chamar no lifespan shutdown do FastAPI."""
    cm = _store_holder.get("cm")
    if cm is not None:
        await cm.__aexit__(None, None, None)  # type: ignore[attr-defined]
    _store_holder["store"] = None
    _store_holder["cm"] = None


def reset_store_for_tests() -> None:
    """Reseta o singleton — uso EXCLUSIVO em testes."""
    _store_holder["store"] = None
    _store_holder["cm"] = None
    _store_holder["lock"] = None
