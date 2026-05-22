"""Integration tests for /api/v1/chat router."""

from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.dependencies import (
    get_current_user,
    get_diagnosis_service,
    get_usage_service,
    require_quota,
)
from app.main import app
from app.shared.enums import FeatureTypeEnum
from tests.conftest import make_user_dto


@pytest.fixture
def mock_diag_svc():
    return AsyncMock()


@pytest.fixture
def mock_usage_svc():
    svc = AsyncMock()
    svc.check_quota = AsyncMock()
    svc.record_usage = AsyncMock()
    return svc


@pytest.fixture
async def client_chat(mock_diag_svc, mock_usage_svc):
    app.dependency_overrides[require_quota(FeatureTypeEnum.CHAT)] = lambda: make_user_dto()
    app.dependency_overrides[get_diagnosis_service] = lambda: mock_diag_svc
    app.dependency_overrides[get_usage_service] = lambda: mock_usage_svc
    app.dependency_overrides[get_current_user] = lambda: make_user_dto()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()


async def test_chat_falls_back_to_text_when_messages_is_not_json(client_chat):
    """Regression for TCC-005: nested except tuple raised TypeError on JSONDecodeError."""
    r = await client_chat.post(
        "/api/v1/chat",
        data={"messages": "ola", "model": "ensemble"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["role"] == "assistant"
    assert isinstance(body["content"], str)
    assert body["content"]
