"""Tests for app/domains/subscriptions/repository.py — SubscriptionRepository."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, call

import pytest

from app.domains.subscriptions.repository import SubscriptionRepository


def _make_orm_plan(**kwargs):
    p = MagicMock()
    p.id = kwargs.get("id", "plan-1")
    p.name = kwargs.get("name", "free")
    p.display_name = kwargs.get("display_name", "Gratuito")
    p.chat_daily_limit = kwargs.get("chat_daily_limit", 10)
    p.inference_daily_limit = kwargs.get("inference_daily_limit", 5)
    p.api_monthly_limit = kwargs.get("api_monthly_limit", 0)
    p.is_active = kwargs.get("is_active", True)
    return p


def _make_orm_sub(plan=None, **kwargs):
    s = MagicMock()
    s.id = kwargs.get("id", "sub-1")
    s.user_id = kwargs.get("user_id", "user-1")
    s.plan_id = kwargs.get("plan_id", "plan-1")
    s.plan = plan or _make_orm_plan()
    s.started_at = kwargs.get("started_at", datetime(2026, 1, 1, tzinfo=UTC))
    s.expires_at = kwargs.get("expires_at", None)
    s.is_active = kwargs.get("is_active", True)
    return s


# ── find_all_active_plans ─────────────────────────────────────────────────────

async def test_find_all_active_plans(mock_db):
    plans = [_make_orm_plan(), _make_orm_plan(name="pro")]
    mock_db.execute.return_value.scalars.return_value.all.return_value = plans
    repo = SubscriptionRepository(mock_db)
    result = await repo.find_all_active_plans()
    assert len(result) == 2
    assert result[0].name == "free"


async def test_find_all_active_plans_empty(mock_db):
    mock_db.execute.return_value.scalars.return_value.all.return_value = []
    repo = SubscriptionRepository(mock_db)
    result = await repo.find_all_active_plans()
    assert result == []


# ── find_plan_by_name ─────────────────────────────────────────────────────────

async def test_find_plan_by_name_found(mock_db):
    mock_db.execute.return_value.scalar_one_or_none.return_value = _make_orm_plan(name="pro")
    repo = SubscriptionRepository(mock_db)
    result = await repo.find_plan_by_name("pro")
    assert result is not None
    assert result.name == "pro"


async def test_find_plan_by_name_not_found(mock_db):
    mock_db.execute.return_value.scalar_one_or_none.return_value = None
    repo = SubscriptionRepository(mock_db)
    result = await repo.find_plan_by_name("unknown")
    assert result is None


# ── find_plan_by_id ───────────────────────────────────────────────────────────

async def test_find_plan_by_id_found(mock_db):
    mock_db.execute.return_value.scalar_one_or_none.return_value = _make_orm_plan(id="plan-99")
    repo = SubscriptionRepository(mock_db)
    result = await repo.find_plan_by_id("plan-99")
    assert result is not None
    assert result.id == "plan-99"


async def test_find_plan_by_id_not_found(mock_db):
    mock_db.execute.return_value.scalar_one_or_none.return_value = None
    repo = SubscriptionRepository(mock_db)
    result = await repo.find_plan_by_id("missing")
    assert result is None


# ── find_user_subscription ────────────────────────────────────────────────────

async def test_find_user_subscription_found(mock_db):
    orm_sub = _make_orm_sub()
    mock_db.execute.return_value.scalar_one_or_none.return_value = orm_sub
    repo = SubscriptionRepository(mock_db)
    result = await repo.find_user_subscription("user-1")
    assert result is not None
    assert result.user_id == "user-1"


async def test_find_user_subscription_not_found(mock_db):
    mock_db.execute.return_value.scalar_one_or_none.return_value = None
    repo = SubscriptionRepository(mock_db)
    result = await repo.find_user_subscription("user-ghost")
    assert result is None


# ── upsert_user_subscription — INSERT (new subscription) ─────────────────────

async def test_upsert_inserts_new_subscription(mock_db):
    orm_sub = _make_orm_sub()
    # First execute: no existing sub → None
    # Second execute (re-fetch with joinedload): returns sub
    mock_db.execute.return_value.scalar_one_or_none.return_value = None
    mock_db.execute.return_value.scalar_one.return_value = orm_sub

    repo = SubscriptionRepository(mock_db)
    result = await repo.upsert_user_subscription("user-1", "plan-1")

    mock_db.add.assert_called_once()
    mock_db.commit.assert_awaited()
    assert result.user_id == "user-1"


# ── upsert_user_subscription — UPDATE (existing subscription) ────────────────

async def test_upsert_updates_existing_subscription(mock_db):
    existing_sub = _make_orm_sub(plan_id="old-plan", is_active=False)
    refreshed_sub = _make_orm_sub(plan_id="plan-new")

    # First call returns existing sub, second call (re-fetch) returns refreshed
    mock_db.execute.return_value.scalar_one_or_none.return_value = existing_sub
    mock_db.execute.return_value.scalar_one.return_value = refreshed_sub

    repo = SubscriptionRepository(mock_db)
    result = await repo.upsert_user_subscription("user-1", "plan-new")

    assert existing_sub.plan_id == "plan-new"
    assert existing_sub.is_active is True
    mock_db.commit.assert_awaited()
    assert result is not None


# ── static _to_dto helpers ────────────────────────────────────────────────────

def test_plan_to_dto():
    orm_plan = _make_orm_plan(name="enterprise", chat_daily_limit=None)
    dto = SubscriptionRepository._plan_to_dto(orm_plan)
    assert dto.name == "enterprise"
    assert dto.chat_daily_limit is None


def test_sub_to_dto():
    orm_sub = _make_orm_sub()
    dto = SubscriptionRepository._sub_to_dto(orm_sub)
    assert dto.user_id == "user-1"
    assert dto.plan.name == "free"
