"""Smoke tests for billing-related routes: subscriptions, usage, action-plans.

These tests verify that:
- Routes are registered and reachable.
- Auth-protected routes reject unauthenticated requests (401).
- Responses carry at least one expected key field.

Intentionally thin — deep assertions live in tests/integration/.
"""

from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.dependencies import (
    get_action_plan_service,
    get_subscription_service,
    get_usage_service,
)
from app.domains.action_plans.schemas import ActionPlanLevelResponse, ActionPlanResponse, SourceResponse
from app.domains.subscriptions.schemas import PlanResponse, SubscriptionResponse
from app.domains.usage.schemas import FeatureUsageResponse, UsageHistoryItemResponse, UsageSummaryResponse
from app.main import app
from app.shared.enums import ActionPlanLevelEnum, FeatureTypeEnum
from tests.conftest import NOW, make_plan_dto, make_subscription_dto
from tests.smoke.conftest import bypass_auth_overrides


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_plan_response() -> PlanResponse:
    p = make_plan_dto()
    return PlanResponse(
        id=p.id,
        name=p.name,
        display_name=p.display_name,
        chat_daily_limit=p.chat_daily_limit,
        inference_daily_limit=p.inference_daily_limit,
        api_monthly_limit=p.api_monthly_limit,
    )


def _make_subscription_response() -> SubscriptionResponse:
    sub = make_subscription_dto()
    return SubscriptionResponse(
        id=sub.id,
        plan=_make_plan_response(),
        started_at=sub.started_at,
        expires_at=sub.expires_at,
        is_active=sub.is_active,
    )


def _make_usage_summary() -> UsageSummaryResponse:
    return UsageSummaryResponse(
        chat=FeatureUsageResponse(used=2, limit=10, remaining=8),
        inference=FeatureUsageResponse(used=1, limit=5, remaining=4),
        api=FeatureUsageResponse(used=0, limit=0, remaining=0),
    )


def _make_action_plan_response() -> ActionPlanResponse:
    return ActionPlanResponse(
        disease_id="ferrugem-asiatica",
        levels=[
            ActionPlanLevelResponse(level=ActionPlanLevelEnum.ESSENCIAL, actions=["Ação 1"])
        ],
        sources=[
            SourceResponse(
                id="src-1",
                name="EMBRAPA",
                detail="Fonte técnica",
                url=None,
                display_order=0,
            )
        ],
    )


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
async def client_subscriptions():
    """Client with subscription service mock + auth bypass."""
    bypass_auth_overrides()
    mock_svc = AsyncMock()
    mock_svc.list_plans = AsyncMock(return_value=[_make_plan_response()])
    mock_svc.get_user_subscription = AsyncMock(return_value=_make_subscription_response())
    mock_svc.subscribe = AsyncMock(return_value=_make_subscription_response())
    app.dependency_overrides[get_subscription_service] = lambda: mock_svc
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()


@pytest.fixture
async def client_usage():
    """Client with usage service mock that has both quota and summary/history methods."""
    bypass_auth_overrides()
    # Override get_usage_service AFTER bypass to provide a mock with get_summary/get_history
    # in addition to the no-op check_quota from bypass_auth_overrides.
    mock_svc = AsyncMock()
    mock_svc.check_quota = AsyncMock()
    mock_svc.record_usage = AsyncMock()
    mock_svc.get_summary = AsyncMock(return_value=_make_usage_summary())
    mock_svc.get_history = AsyncMock(
        return_value=[
            UsageHistoryItemResponse(
                id="log-1",
                feature=FeatureTypeEnum.INFERENCE,
                used_at=NOW,
            )
        ]
    )
    app.dependency_overrides[get_usage_service] = lambda: mock_svc
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()


@pytest.fixture
async def client_action_plans():
    """Client with action-plan service mock + auth bypass."""
    bypass_auth_overrides()
    mock_svc = AsyncMock()
    mock_svc.get_by_disease = AsyncMock(return_value=_make_action_plan_response())
    mock_svc.get_level = AsyncMock(
        return_value=ActionPlanLevelResponse(
            level=ActionPlanLevelEnum.ESSENCIAL, actions=["Ação 1"]
        )
    )
    app.dependency_overrides[get_action_plan_service] = lambda: mock_svc
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()


# ── Negative / 401 ────────────────────────────────────────────────────────────

async def test_subscriptions_me_requires_auth(smoke_client):
    """GET /subscriptions/me without auth must return 401."""
    r = await smoke_client.get("/api/v1/subscriptions/me")
    assert r.status_code == 401


# ── /api/v1/subscriptions/plans (public) ─────────────────────────────────────

async def test_list_plans_no_auth_200(client_subscriptions):
    """GET /plans is public — no Authorization header needed → 200."""
    r = await client_subscriptions.get("/api/v1/subscriptions/plans")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
    assert data[0]["name"] == "free"


# ── /api/v1/subscriptions/me ──────────────────────────────────────────────────

async def test_get_my_subscription_200(client_subscriptions):
    r = await client_subscriptions.get("/api/v1/subscriptions/me")
    assert r.status_code == 200
    data = r.json()
    assert "plan" in data
    assert data["is_active"] is True


async def test_subscribe_201(client_subscriptions):
    r = await client_subscriptions.post(
        "/api/v1/subscriptions/me", json={"plan_name": "free"}
    )
    assert r.status_code == 201
    data = r.json()
    assert "id" in data


# ── /api/v1/usage/me ──────────────────────────────────────────────────────────

async def test_get_usage_summary_200(client_usage):
    r = await client_usage.get("/api/v1/usage/me")
    assert r.status_code == 200
    data = r.json()
    assert "chat" in data
    assert "inference" in data


async def test_get_usage_history_200(client_usage):
    r = await client_usage.get("/api/v1/usage/me/history")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
    assert data[0]["feature"] == "inference"


# ── /api/v1/action-plans ──────────────────────────────────────────────────────

async def test_get_action_plan_200(client_action_plans):
    r = await client_action_plans.get("/api/v1/action-plans/ferrugem-asiatica")
    assert r.status_code == 200
    data = r.json()
    assert data["disease_id"] == "ferrugem-asiatica"
    assert "levels" in data


async def test_get_action_plan_level_200(client_action_plans):
    r = await client_action_plans.get("/api/v1/action-plans/ferrugem-asiatica/essencial")
    assert r.status_code == 200
    data = r.json()
    assert data["level"] == "essencial"
    assert isinstance(data["actions"], list)
