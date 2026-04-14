"""Tests for app/domains/subscriptions/service.py — SubscriptionService."""

from unittest.mock import AsyncMock

import pytest

from app.core.exceptions import NotFoundError
from app.domains.subscriptions.schemas import SubscribeRequest
from app.domains.subscriptions.service import SubscriptionService
from tests.conftest import make_plan_dto, make_subscription_dto


@pytest.fixture
def sub_repo():
    repo = AsyncMock()
    repo.find_all_active_plans = AsyncMock(return_value=[make_plan_dto()])
    repo.find_plan_by_name = AsyncMock(return_value=make_plan_dto())
    repo.find_user_subscription = AsyncMock(return_value=None)
    repo.upsert_user_subscription = AsyncMock(return_value=make_subscription_dto())
    return repo


# ── list_plans ────────────────────────────────────────────────────────────────

async def test_list_plans_returns_list(sub_repo):
    svc = SubscriptionService(sub_repo)
    result = await svc.list_plans()
    assert len(result) == 1
    assert result[0].name == "free"


async def test_list_plans_empty(sub_repo):
    sub_repo.find_all_active_plans.return_value = []
    svc = SubscriptionService(sub_repo)
    result = await svc.list_plans()
    assert result == []


# ── get_user_subscription ─────────────────────────────────────────────────────

async def test_get_user_subscription_none(sub_repo):
    sub_repo.find_user_subscription.return_value = None
    svc = SubscriptionService(sub_repo)
    result = await svc.get_user_subscription("user-1")
    assert result is None


async def test_get_user_subscription_found(sub_repo):
    sub_repo.find_user_subscription.return_value = make_subscription_dto()
    svc = SubscriptionService(sub_repo)
    result = await svc.get_user_subscription("user-1")
    assert result is not None
    assert result.plan.name == "free"


# ── subscribe ─────────────────────────────────────────────────────────────────

async def test_subscribe_success(sub_repo):
    svc = SubscriptionService(sub_repo)
    result = await svc.subscribe("user-1", SubscribeRequest(plan_name="free"))
    assert result is not None
    sub_repo.upsert_user_subscription.assert_awaited_once()


async def test_subscribe_plan_not_found(sub_repo):
    sub_repo.find_plan_by_name.return_value = None
    svc = SubscriptionService(sub_repo)
    with pytest.raises(NotFoundError, match="Plan"):
        await svc.subscribe("user-1", SubscribeRequest(plan_name="nonexistent"))


# ── static helpers ────────────────────────────────────────────────────────────

def test_plan_to_response():
    plan = make_plan_dto(chat_daily_limit=None)
    result = SubscriptionService._plan_to_response(plan)
    assert result.name == "free"
    assert result.chat_daily_limit is None


def test_sub_to_response():
    sub = make_subscription_dto()
    result = SubscriptionService._sub_to_response(sub)
    assert result.is_active is True
    assert result.plan.name == "free"
