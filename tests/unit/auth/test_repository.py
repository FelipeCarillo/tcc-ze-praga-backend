"""Tests for app/domains/auth/repository.py — UserRepository."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.domains.auth.dto import UserCreateDTO
from app.domains.auth.repository import UserRepository


def _make_orm_user(**kwargs):
    user = MagicMock()
    user.id = kwargs.get("id", "user-1")
    user.email = kwargs.get("email", "test@test.com")
    user.password_hash = kwargs.get("password_hash", "$2b$hashed")
    user.full_name = kwargs.get("full_name", "Test")
    user.is_active = kwargs.get("is_active", True)
    user.created_at = kwargs.get("created_at", datetime(2026, 1, 1, tzinfo=UTC))
    user.updated_at = kwargs.get("updated_at", datetime(2026, 1, 1, tzinfo=UTC))
    return user


# ── find_by_email ─────────────────────────────────────────────────────────────

async def test_find_by_email_found(mock_db):
    orm_user = _make_orm_user()
    mock_db.execute.return_value.scalar_one_or_none.return_value = orm_user
    repo = UserRepository(mock_db)
    result = await repo.find_by_email("test@test.com")
    assert result is not None
    assert result.email == "test@test.com"


async def test_find_by_email_not_found(mock_db):
    mock_db.execute.return_value.scalar_one_or_none.return_value = None
    repo = UserRepository(mock_db)
    result = await repo.find_by_email("ghost@test.com")
    assert result is None


# ── find_by_id ────────────────────────────────────────────────────────────────

async def test_find_by_id_found(mock_db):
    orm_user = _make_orm_user(id="user-abc")
    mock_db.execute.return_value.scalar_one_or_none.return_value = orm_user
    repo = UserRepository(mock_db)
    result = await repo.find_by_id("user-abc")
    assert result is not None
    assert result.id == "user-abc"


async def test_find_by_id_not_found(mock_db):
    mock_db.execute.return_value.scalar_one_or_none.return_value = None
    repo = UserRepository(mock_db)
    result = await repo.find_by_id("missing")
    assert result is None


# ── create ────────────────────────────────────────────────────────────────────

async def test_create_returns_dto(mock_db):
    orm_user = _make_orm_user(email="new@test.com")

    async def fake_refresh(obj):
        obj.id = orm_user.id
        obj.email = orm_user.email
        obj.password_hash = orm_user.password_hash
        obj.full_name = orm_user.full_name
        obj.is_active = orm_user.is_active
        obj.created_at = orm_user.created_at
        obj.updated_at = orm_user.updated_at

    mock_db.refresh.side_effect = fake_refresh
    mock_db.execute.return_value.scalar_one_or_none.return_value = orm_user

    repo = UserRepository(mock_db)
    dto = await repo.create(UserCreateDTO(email="new@test.com", password_hash="h", full_name="N"))

    mock_db.add.assert_called_once()
    mock_db.commit.assert_awaited_once()
    assert dto.email == "new@test.com"


# ── update ────────────────────────────────────────────────────────────────────

async def test_update_user_found(mock_db):
    orm_user = _make_orm_user()
    mock_db.execute.return_value.scalar_one_or_none.return_value = orm_user
    repo = UserRepository(mock_db)
    result = await repo.update("user-1", full_name="New Name")
    assert result is not None
    mock_db.commit.assert_awaited()


async def test_update_user_not_found(mock_db):
    mock_db.execute.return_value.scalar_one_or_none.return_value = None
    repo = UserRepository(mock_db)
    result = await repo.update("missing", full_name="X")
    assert result is None


# ── soft_delete ───────────────────────────────────────────────────────────────

async def test_soft_delete_calls_update(mock_db):
    orm_user = _make_orm_user()
    mock_db.execute.return_value.scalar_one_or_none.return_value = orm_user
    repo = UserRepository(mock_db)
    await repo.soft_delete("user-1")
    mock_db.commit.assert_awaited()


# ── _to_dto ───────────────────────────────────────────────────────────────────

def test_to_dto_mapping():
    orm_user = _make_orm_user(
        id="u1",
        email="a@b.com",
        password_hash="hash",
        full_name="Alice",
        is_active=False,
    )
    dto = UserRepository._to_dto(orm_user)
    assert dto.id == "u1"
    assert dto.email == "a@b.com"
    assert dto.full_name == "Alice"
    assert dto.is_active is False
