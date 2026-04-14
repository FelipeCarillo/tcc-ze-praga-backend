"""Tests for app/domains/usage/repository.py — UsageRepository."""

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from app.domains.usage.repository import UsageRepository
from app.shared.enums import FeatureTypeEnum


def _make_orm_log(**kwargs):
    log = MagicMock()
    log.id = kwargs.get("id", "log-1")
    log.user_id = kwargs.get("user_id", "user-1")
    log.feature = kwargs.get("feature", "inference")
    log.used_at = kwargs.get("used_at", datetime(2026, 4, 13, tzinfo=UTC))
    log.metadata_ = kwargs.get("metadata_", None)
    return log


# ── count_today ───────────────────────────────────────────────────────────────

async def test_count_today(mock_db):
    mock_db.execute.return_value.scalar_one.return_value = 3
    repo = UsageRepository(mock_db)
    count = await repo.count_today("user-1", FeatureTypeEnum.INFERENCE)
    assert count == 3


# ── count_this_month ──────────────────────────────────────────────────────────

async def test_count_this_month(mock_db):
    mock_db.execute.return_value.scalar_one.return_value = 42
    repo = UsageRepository(mock_db)
    count = await repo.count_this_month("user-1", FeatureTypeEnum.API)
    assert count == 42


# ── record ────────────────────────────────────────────────────────────────────

async def test_record_without_metadata(mock_db):
    orm_log = _make_orm_log()

    async def fake_refresh(obj):
        obj.id = orm_log.id
        obj.user_id = orm_log.user_id
        obj.feature = orm_log.feature
        obj.used_at = orm_log.used_at
        obj.metadata_ = None

    mock_db.refresh.side_effect = fake_refresh
    repo = UsageRepository(mock_db)
    dto = await repo.record("user-1", FeatureTypeEnum.INFERENCE)
    mock_db.add.assert_called_once()
    mock_db.commit.assert_awaited_once()
    assert dto.feature == FeatureTypeEnum.INFERENCE


async def test_record_with_metadata(mock_db):
    orm_log = _make_orm_log(metadata_={"disease_id": "ferrugem"})

    async def fake_refresh(obj):
        obj.id = orm_log.id
        obj.user_id = orm_log.user_id
        obj.feature = orm_log.feature
        obj.used_at = orm_log.used_at
        obj.metadata_ = orm_log.metadata_

    mock_db.refresh.side_effect = fake_refresh
    repo = UsageRepository(mock_db)
    dto = await repo.record("user-1", FeatureTypeEnum.INFERENCE, {"disease_id": "ferrugem"})
    assert dto.metadata == {"disease_id": "ferrugem"}


# ── find_recent ───────────────────────────────────────────────────────────────

async def test_find_recent(mock_db):
    logs = [_make_orm_log(id=f"log-{i}") for i in range(3)]
    mock_db.execute.return_value.scalars.return_value.all.return_value = logs
    repo = UsageRepository(mock_db)
    result = await repo.find_recent("user-1", limit=50)
    assert len(result) == 3


async def test_find_recent_empty(mock_db):
    mock_db.execute.return_value.scalars.return_value.all.return_value = []
    repo = UsageRepository(mock_db)
    result = await repo.find_recent("user-1")
    assert result == []


# ── _to_dto ───────────────────────────────────────────────────────────────────

def test_to_dto_with_none_metadata():
    orm_log = _make_orm_log(metadata_=None)
    dto = UsageRepository._to_dto(orm_log)
    assert dto.metadata is None
    assert dto.feature == FeatureTypeEnum.INFERENCE


def test_to_dto_with_metadata():
    orm_log = _make_orm_log(metadata_={"key": "val"})
    dto = UsageRepository._to_dto(orm_log)
    assert dto.metadata == {"key": "val"}
