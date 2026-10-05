"""Checkpointer e pool do LangGraph — conexão que se recupera sozinha."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.db import checkpointer as cp_module
from app.db import langgraph_pool


@pytest.fixture(autouse=True)
def _reset():
    cp_module.reset_checkpointer_for_tests()
    yield
    cp_module.reset_checkpointer_for_tests()


async def test_get_checkpointer_usa_pool_e_reaproveita(monkeypatch):
    fake_pool = AsyncMock(name="pool")
    open_pool = AsyncMock(return_value=fake_pool)
    monkeypatch.setattr("app.db.langgraph_pool.open_langgraph_pool", open_pool)
    fake_saver = AsyncMock(name="saver")
    fake_cls = MagicMock(return_value=fake_saver)
    fake_module = MagicMock(AsyncPostgresSaver=fake_cls)

    with patch.dict("sys.modules", {"langgraph.checkpoint.postgres.aio": fake_module}):
        first = await cp_module.get_checkpointer()
        second = await cp_module.get_checkpointer()

    assert first is second is fake_saver
    open_pool.assert_awaited_once()
    fake_cls.assert_called_once_with(conn=fake_pool)
    fake_saver.setup.assert_awaited_once()

    await cp_module.close_checkpointer()
    fake_pool.close.assert_awaited_once()
    assert cp_module._holder["checkpointer"] is None


async def test_open_langgraph_pool_testa_a_conexao_antes_de_entregar(monkeypatch):
    """O ``check`` é o que troca uma conexão derrubada por uma nova."""
    created = {}

    class FakePool:
        check_connection = object()

        def __init__(self, conninfo, **kwargs):
            created.update(conninfo=conninfo, **kwargs)
            self.open = AsyncMock()

    monkeypatch.setattr("psycopg_pool.AsyncConnectionPool", FakePool)

    pool = await langgraph_pool.open_langgraph_pool("postgresql://x")

    assert created["check"] is FakePool.check_connection
    assert created["kwargs"]["autocommit"] is True
    assert created["kwargs"]["prepare_threshold"] == 0
    assert created["max_size"] == langgraph_pool.POOL_MAX_SIZE
    assert created["open"] is False
    pool.open.assert_awaited_once()
