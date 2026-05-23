"""Unit tests for ApiKeyRepository (mocked DB)."""

from datetime import UTC, datetime
from unittest.mock import MagicMock

from app.domains.auth.api_key_dto import ApiKeyCreateDTO
from app.domains.auth.api_key_repository import ApiKeyRepository


NOW = datetime(2026, 5, 23, tzinfo=UTC)


def _make_orm(**kwargs):
    row = MagicMock()
    row.id = kwargs.get("id", "apik-1")
    row.user_id = kwargs.get("user_id", "user-1")
    row.name = kwargs.get("name", "my-key")
    row.key_hash = kwargs.get("key_hash", "$2b$12$hash")
    row.key_prefix = kwargs.get("key_prefix", "zp_live_abcd")
    row.scopes = kwargs.get("scopes", ["diagnoses:analyze"])
    row.is_active = kwargs.get("is_active", True)
    row.last_used_at = kwargs.get("last_used_at", None)
    row.created_at = kwargs.get("created_at", NOW)
    row.revoked_at = kwargs.get("revoked_at", None)
    return row


# ── create ────────────────────────────────────────────────────────────────────

async def test_create_returns_dto(mock_db):
    orm_row = _make_orm()

    async def fake_refresh(obj):
        obj.id = orm_row.id
        obj.user_id = orm_row.user_id
        obj.name = orm_row.name
        obj.key_hash = orm_row.key_hash
        obj.key_prefix = orm_row.key_prefix
        obj.scopes = orm_row.scopes
        obj.is_active = orm_row.is_active
        obj.last_used_at = None
        obj.created_at = orm_row.created_at
        obj.revoked_at = None

    mock_db.refresh.side_effect = fake_refresh

    repo = ApiKeyRepository(mock_db)
    dto = await repo.create(
        ApiKeyCreateDTO(
            user_id="user-1",
            name="my-key",
            key_hash="$2b$12$hash",
            key_prefix="zp_live_abcd",
            scopes=["diagnoses:analyze"],
        )
    )

    mock_db.add.assert_called_once()
    mock_db.commit.assert_awaited_once()
    assert dto.id == "apik-1"
    assert dto.user_id == "user-1"
    assert dto.scopes == ["diagnoses:analyze"]


# ── find_active_by_user ───────────────────────────────────────────────────────

async def test_find_active_by_user_returns_list(mock_db):
    orm_rows = [_make_orm(id="k1"), _make_orm(id="k2")]
    scalars = MagicMock()
    scalars.all.return_value = orm_rows
    mock_db.execute.return_value.scalars.return_value = scalars

    repo = ApiKeyRepository(mock_db)
    dtos = await repo.find_active_by_user("user-1")
    assert [d.id for d in dtos] == ["k1", "k2"]


# ── find_by_id ────────────────────────────────────────────────────────────────

async def test_find_by_id_scoped_by_user(mock_db):
    orm = _make_orm()
    mock_db.execute.return_value.scalar_one_or_none.return_value = orm
    repo = ApiKeyRepository(mock_db)
    dto = await repo.find_by_id("apik-1", "user-1")
    assert dto is not None
    assert dto.id == "apik-1"


async def test_find_by_id_not_found(mock_db):
    mock_db.execute.return_value.scalar_one_or_none.return_value = None
    repo = ApiKeyRepository(mock_db)
    dto = await repo.find_by_id("ghost", "user-1")
    assert dto is None


# ── find_by_prefix_active ─────────────────────────────────────────────────────

async def test_find_by_prefix_active_returns_candidates(mock_db):
    orm_rows = [_make_orm(id="k1"), _make_orm(id="k2")]
    scalars = MagicMock()
    scalars.all.return_value = orm_rows
    mock_db.execute.return_value.scalars.return_value = scalars

    repo = ApiKeyRepository(mock_db)
    dtos = await repo.find_by_prefix_active("zp_live_abcd")
    assert len(dtos) == 2


# ── revoke ────────────────────────────────────────────────────────────────────

async def test_revoke_existing_returns_true(mock_db):
    orm = _make_orm(is_active=True)
    mock_db.execute.return_value.scalar_one_or_none.return_value = orm
    repo = ApiKeyRepository(mock_db)
    ok = await repo.revoke("apik-1", "user-1")
    assert ok is True
    assert orm.is_active is False
    assert orm.revoked_at is not None
    mock_db.commit.assert_awaited_once()


async def test_revoke_missing_returns_false(mock_db):
    mock_db.execute.return_value.scalar_one_or_none.return_value = None
    repo = ApiKeyRepository(mock_db)
    ok = await repo.revoke("ghost", "user-1")
    assert ok is False
    mock_db.commit.assert_not_called()


# ── touch_last_used ───────────────────────────────────────────────────────────

async def test_touch_last_used_sets_timestamp(mock_db):
    orm = _make_orm(last_used_at=None)
    mock_db.execute.return_value.scalar_one_or_none.return_value = orm
    repo = ApiKeyRepository(mock_db)
    await repo.touch_last_used("apik-1")
    assert orm.last_used_at is not None
    mock_db.commit.assert_awaited_once()


async def test_touch_last_used_noop_when_missing(mock_db):
    mock_db.execute.return_value.scalar_one_or_none.return_value = None
    repo = ApiKeyRepository(mock_db)
    await repo.touch_last_used("ghost")
    mock_db.commit.assert_not_called()
