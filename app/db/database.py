from collections.abc import AsyncGenerator
from typing import Any
from uuid import uuid4

from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import settings

# Porta do pooler do Supabase (Supavisor) em transaction mode.
TRANSACTION_POOLER_PORT = 6543


def engine_options(url: str) -> tuple[str, dict[str, Any]]:
    """URL e kwargs do engine conforme o modo do pooler (TCC-104).

    O teste de carga de 18/10/2026 deu 500 a partir de 15 usuários
    (``EMAXCONNSESSION``): em session mode (5432) o Supabase dá uma conexão do
    banco a cada cliente e para em 15. Em transaction mode (6543) as conexões
    são compartilhadas entre muitos clientes, mas o pooler não suporta prepared
    statements — o asyncpg e o SQLAlchemy cacheiam os deles, então os dois
    caches são desligados e cada statement ganha nome único (receita do
    SQLAlchemy para PgBouncer).
    """
    kwargs: dict[str, Any] = {
        "echo": settings.is_development,
        "pool_pre_ping": True,
        "pool_size": settings.db_pool_size,
        "max_overflow": settings.db_max_overflow,
    }
    parsed = make_url(url)
    if parsed.port == TRANSACTION_POOLER_PORT:
        url = parsed.update_query_dict({"prepared_statement_cache_size": "0"}).render_as_string(
            hide_password=False
        )
        kwargs["connect_args"] = {
            "statement_cache_size": 0,
            "prepared_statement_name_func": lambda: f"__asyncpg_{uuid4()}__",
        }
    return url, kwargs


_url, _kwargs = engine_options(settings.database_url)
engine = create_async_engine(_url, **_kwargs)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session
