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
        store_module.settings, "openai_embeddings_model", "test-embed-model"
    )
    monkeypatch.setattr(
        store_module.settings, "openai_embeddings_dims", 1024
    )
    monkeypatch.setattr(
        store_module.settings, "openai_api_key", "sk-test"
    )

    fake_embeddings_instance = MagicMock(name="OpenAIEmbeddings()")
    with patch(
        "langchain_openai.OpenAIEmbeddings",
        return_value=fake_embeddings_instance,
    ) as embed_cls:
        cfg = store_module._build_index_config()

    embed_cls.assert_called_once_with(
        model="test-embed-model", api_key="sk-test"
    )
    assert cfg["dims"] == 1024
    assert cfg["embed"] is fake_embeddings_instance
    assert cfg["fields"] == ["summary_text"]


async def test_get_store_caches_singleton(monkeypatch):
    """Primeira chamada cria o store via from_conn_string + setup(); segunda reusa."""
    monkeypatch.setattr(
        store_module,
        "_build_index_config",
        lambda: {"dims": 1536, "embed": MagicMock(), "fields": ["summary_text"]},
    )

    fake_store = AsyncMock(name="AsyncPostgresStore-instance")

    # Simula o async context manager retornado por from_conn_string.
    fake_cm = AsyncMock(name="from_conn_string-cm")
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
        result1 = await store_module.get_store()
        result2 = await store_module.get_store()

    assert result1 is fake_store
    assert result2 is fake_store
    # setup() so chama 1x; from_conn_string so chama 1x
    fake_store.setup.assert_awaited_once()
    fake_store_cls.from_conn_string.assert_called_once()


async def test_close_store_resets_singleton(monkeypatch):
    """Apos close_store(), singleton volta a None e proxima chamada reinstancia."""
    fake_store = AsyncMock(name="store")
    fake_cm = AsyncMock(name="cm")
    fake_cm.__aenter__ = AsyncMock(return_value=fake_store)
    fake_cm.__aexit__ = AsyncMock(return_value=False)

    store_module._store_holder["store"] = fake_store
    store_module._store_holder["cm"] = fake_cm

    await store_module.close_store()

    assert store_module._store_holder["store"] is None
    assert store_module._store_holder["cm"] is None
    fake_cm.__aexit__.assert_awaited_once_with(None, None, None)


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
