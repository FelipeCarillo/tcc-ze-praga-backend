"""Integration tests for /api/v1/subscriptions router."""

from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.dependencies import get_current_user, get_subscription_service
from app.core.exceptions import NotFoundError
from app.main import app
from tests.conftest import make_user_dto
from tests.integration.conftest import make_plan_response, make_subscription_response


@pytest.fixture
def mock_sub_svc():
    svc = AsyncMock()
    svc.list_plans = AsyncMock(return_value=[make_plan_response()])
    svc.get_user_subscription = AsyncMock(return_value=make_subscription_response())
    svc.subscribe = AsyncMock(return_value=make_subscription_response())
    return svc


@pytest.fixture
async def client_sub(mock_sub_svc):
    app.dependency_overrides[get_subscription_service] = lambda: mock_sub_svc
    app.dependency_overrides[get_current_user] = lambda: make_user_dto()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()


async def test_list_plans_200(client_sub):
    r = await client_sub.get("/api/v1/subscriptions/plans")
    assert r.status_code == 200
    assert len(r.json()) == 1


async def test_get_my_subscription_200(client_sub):
    r = await client_sub.get("/api/v1/subscriptions/me")
    assert r.status_code == 200
    assert r.json()["plan"]["name"] == "free"


async def test_get_my_subscription_none(client_sub, mock_sub_svc):
    mock_sub_svc.get_user_subscription.return_value = None
    r = await client_sub.get("/api/v1/subscriptions/me")
    assert r.status_code == 200
    assert r.json() is None


async def test_subscribe_201(client_sub):
    r = await client_sub.post(
        "/api/v1/subscriptions/me", json={"plan_name": "free"}
    )
    assert r.status_code == 201


async def test_subscribe_plan_not_found(client_sub, mock_sub_svc):
    mock_sub_svc.subscribe.side_effect = NotFoundError("Plan", "unknown")
    r = await client_sub.post(
        "/api/v1/subscriptions/me", json={"plan_name": "unknown"}
    )
    assert r.status_code == 404
