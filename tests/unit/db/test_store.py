"""Testes do wrapper de Store (TCC-044).

Mocka o AsyncPostgresStore + OpenAIEmbeddings — nao bate em DB real.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.db import store as store_module


@pytest.fixture(autouse=True)
def reset_store_singleton():
    """Reseta o singleton antes/depois de cada teste pra isolamento."""
    store_module.reset_store_for_tests()
    yield
    store_module.reset_store_for_tests()


def test_make_conn_string_strips_asyncpg_driver(monkeypatch):
    monkeypatch.setattr(
        store_module.settings,
        "database_url",
        "postgresql+asyncpg://user:pass@host:5432/db",
    )
    assert (
        store_module._make_conn_string()
        == "postgresql://user:pass@host:5432/db"
    )


def test_make_conn_string_strips_postgres_asyncpg(monkeypatch):
    monkeypatch.setattr(
        store_module.settings,
        "database_url",
        "postgres+asyncpg://user:pass@host:5432/db",
    )
    assert (
        store_module._make_conn_string()
        == "postgresql://user:pass@host:5432/db"
    )


def test_make_conn_string_passes_through_clean_url(monkeypatch):
    monkeypatch.setattr(
        store_module.settings,
        "database_url",
        "postgresql://user:pass@host:5432/db",
    )
    assert (
        store_module._make_conn_string()
        == "postgresql://user:pass@host:5432/db"
    )


def test_build_index_config_uses_settings(monkeypatch):
    monkeypatch.setattr(
        store_module.settings, "embeddings_model", "google_genai:test-embed"
    )
    monkeypatch.setattr(store_module.settings, "embeddings_dims", 1024)

    fake_embeddings_instance = MagicMock(name="Embeddings()")
    with patch(
        "app.core.llm.get_embeddings",
        return_value=fake_embeddings_instance,
    ) as factory:
        cfg = store_module._build_index_config()

    # Embeddings saem da factory agnóstica — qualquer provider do LangChain.
    factory.assert_called_once_with("google_genai:test-embed")
    assert cfg["dims"] == 1024
    assert cfg["embed"] is fake_embeddings_instance
    assert cfg["fields"] == ["summary_text"]


async def test_get_store_caches_singleton(monkeypatch):
    """Primeira chamada abre o pool + setup(); segunda reusa o mesmo store."""
    monkeypatch.setattr(
        store_module,
        "_build_index_config",
        lambda: {"dims": 1536, "embed": MagicMock(), "fields": ["summary_text"]},
    )
    fake_pool = AsyncMock(name="pool")
    open_pool = AsyncMock(return_value=fake_pool)
    monkeypatch.setattr("app.db.langgraph_pool.open_langgraph_pool", open_pool)

    fake_store = AsyncMock(name="AsyncPostgresStore-instance")
    fake_store_cls = MagicMock(return_value=fake_store)
    fake_module = MagicMock()
    fake_module.AsyncPostgresStore = fake_store_cls

    with patch.dict(
        "sys.modules",
        {"langgraph.store.postgres.aio": fake_module},
    ):
        result1 = await store_module.get_store()
        result2 = await store_module.get_store()

    assert result1 is fake_store
    assert result2 is fake_store
    # Pool (não conexão única): sobrevive a conexão derrubada pelo servidor.
    open_pool.assert_awaited_once()
    assert fake_store_cls.call_args.kwargs["conn"] is fake_pool
    fake_store.setup.assert_awaited_once()
    assert store_module._store_holder["pool"] is fake_pool


async def test_close_store_resets_singleton(monkeypatch):
    """Apos close_store(), singleton volta a None e o pool é fechado."""
    fake_store = AsyncMock(name="store")
    fake_pool = AsyncMock(name="pool")

    store_module._store_holder["store"] = fake_store
    store_module._store_holder["pool"] = fake_pool

    await store_module.close_store()

    assert store_module._store_holder["store"] is None
    assert store_module._store_holder["pool"] is None
    fake_pool.close.assert_awaited_once()


async def test_close_store_noop_when_not_initialized():
    """close_store() em singleton nao inicializado nao explode."""
    await store_module.close_store()  # sem patch — singleton vazio


async def test_open_store_context_manager(monkeypatch):
    """open_store() abre store, chama setup, e fecha no __aexit__."""
    monkeypatch.setattr(
        store_module,
        "_build_index_config",
        lambda: {"dims": 1536, "embed": MagicMock(), "fields": ["summary_text"]},
    )

    fake_store = AsyncMock(name="store")
    fake_cm = AsyncMock(name="cm")
    fake_cm.__aenter__ = AsyncMock(return_value=fake_store)
    fake_cm.__aexit__ = AsyncMock(return_value=False)

    fake_store_cls = MagicMock()
    fake_store_cls.from_conn_string = MagicMock(return_value=fake_cm)

    fake_module = MagicMock()
    fake_module.AsyncPostgresStore = fake_store_cls

    with patch.dict(
        "sys.modules",
        {"langgraph.store.postgres.aio": fake_module},
    ):
        async with store_module.open_store() as s:
            assert s is fake_store

    fake_store.setup.assert_awaited_once()
    fake_cm.__aexit__.assert_awaited_once()
