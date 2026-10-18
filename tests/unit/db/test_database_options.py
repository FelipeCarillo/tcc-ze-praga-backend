"""TCC-104 — engine do SQLAlchemy conforme o modo do pooler do Supabase."""

from __future__ import annotations

from sqlalchemy.engine import make_url

from app.config import settings
from app.db import checkpointer as checkpointer_module
from app.db import store as store_module
from app.db.database import engine_options

SESSION = "postgresql+asyncpg://postgres.ref:se%40nha@aws-0-us-east-1.pooler.supabase.com:5432/postgres"
TRANSACTION = SESSION.replace(":5432/", ":6543/")


def test_session_mode_nao_mexe_em_prepared_statements():
    url, kwargs = engine_options(SESSION)
    assert url == SESSION
    assert "connect_args" not in kwargs
    assert kwargs["pool_size"] == settings.db_pool_size
    assert kwargs["max_overflow"] == settings.db_max_overflow
    assert kwargs["pool_pre_ping"] is True


def test_transaction_mode_desliga_os_caches_de_prepared_statement():
    url, kwargs = engine_options(TRANSACTION)
    parsed = make_url(url)
    assert parsed.port == 6543
    assert parsed.query["prepared_statement_cache_size"] == "0"
    assert parsed.password == "se@nha"  # senha preservada (não vira ***)
    args = kwargs["connect_args"]
    assert args["statement_cache_size"] == 0
    a, b = args["prepared_statement_name_func"](), args["prepared_statement_name_func"]()
    assert a != b and a.startswith("__asyncpg_")


def test_pool_configuravel(monkeypatch):
    monkeypatch.setattr(settings, "db_pool_size", 3)
    monkeypatch.setattr(settings, "db_max_overflow", 1)
    _, kwargs = engine_options(SESSION)
    assert (kwargs["pool_size"], kwargs["max_overflow"]) == (3, 1)


def test_checkpointer_e_store_usam_a_url_de_session(monkeypatch):
    monkeypatch.setattr(settings, "database_url", TRANSACTION)
    monkeypatch.setattr(settings, "database_session_url", SESSION)
    esperado = SESSION.replace("postgresql+asyncpg://", "postgresql://")
    assert checkpointer_module._make_conn_string() == esperado
    assert store_module._make_conn_string() == esperado


def test_sem_url_de_session_cai_no_database_url(monkeypatch):
    monkeypatch.setattr(settings, "database_url", SESSION)
    monkeypatch.setattr(settings, "database_session_url", None)
    assert settings.session_database_url == SESSION
    assert checkpointer_module._make_conn_string().endswith(":5432/postgres")
