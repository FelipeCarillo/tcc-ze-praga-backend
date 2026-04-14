"""Integration tests for /api/v1/usage router."""

from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.dependencies import get_current_user, get_usage_service
from app.main import app
from tests.conftest import make_user_dto, make_usage_log_dto, NOW
from tests.integration.conftest import make_usage_summary
from app.domains.usage.schemas import UsageHistoryItemResponse
from app.shared.enums import FeatureTypeEnum


@pytest.fixture
def mock_usage_svc():
    svc = AsyncMock()
    svc.get_summary = AsyncMock(return_value=make_usage_summary())
    svc.get_history = AsyncMock(
        return_value=[
            UsageHistoryItemResponse(
                id="log-1", feature=FeatureTypeEnum.INFERENCE, used_at=NOW
            )
        ]
    )
    return svc


@pytest.fixture
async def client_usage(mock_usage_svc):
    app.dependency_overrides[get_usage_service] = lambda: mock_usage_svc
    app.dependency_overrides[get_current_user] = lambda: make_user_dto()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()


async def test_get_usage_summary_200(client_usage):
    r = await client_usage.get("/api/v1/usage/me")
    assert r.status_code == 200
    data = r.json()
    assert "chat" in data
    assert "inference" in data
    assert "api" in data


async def test_get_usage_history_200(client_usage):
    r = await client_usage.get("/api/v1/usage/me/history")
    assert r.status_code == 200
    assert len(r.json()) == 1


async def test_get_usage_history_with_limit(client_usage, mock_usage_svc):
    r = await client_usage.get("/api/v1/usage/me/history?limit=10")
    assert r.status_code == 200
    mock_usage_svc.get_history.assert_awaited_with("user-uuid-1", 10)
