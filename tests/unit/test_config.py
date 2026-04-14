"""Tests for app/config.py — Settings properties."""

from app.config import Settings


def test_allowed_origins_list_single():
    s = Settings(
        database_url="postgresql+asyncpg://x",
        supabase_url="https://x.supabase.co",
        supabase_service_role_key="key",
        jwt_secret_key="secret",
        allowed_origins="http://localhost:3000",
    )
    assert s.allowed_origins_list == ["http://localhost:3000"]


def test_allowed_origins_list_multiple():
    s = Settings(
        database_url="postgresql+asyncpg://x",
        supabase_url="https://x.supabase.co",
        supabase_service_role_key="key",
        jwt_secret_key="secret",
        allowed_origins="http://localhost:3000,https://zepraga.com.br",
    )
    assert s.allowed_origins_list == ["http://localhost:3000", "https://zepraga.com.br"]


def test_is_development_true():
    s = Settings(
        database_url="postgresql+asyncpg://x",
        supabase_url="https://x.supabase.co",
        supabase_service_role_key="key",
        jwt_secret_key="secret",
        app_env="development",
    )
    assert s.is_development is True


def test_is_development_false():
    s = Settings(
        database_url="postgresql+asyncpg://x",
        supabase_url="https://x.supabase.co",
        supabase_service_role_key="key",
        jwt_secret_key="secret",
        app_env="production",
    )
    assert s.is_development is False
