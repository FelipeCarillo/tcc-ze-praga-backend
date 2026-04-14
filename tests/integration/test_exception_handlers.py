"""Integration tests for exception handlers in main.py and get_current_user in dependencies.py."""

from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.dependencies import get_current_user, get_usage_service, require_quota
from app.core.exceptions import QuotaExceededError
from app.main import app
from app.shared.enums import FeatureTypeEnum
from tests.conftest import make_user_dto
from tests.integration.conftest import make_usage_summary


# ── get_current_user integration ──────────────────────────────────────────────

async def test_missing_token_returns_401():
    """Calling a protected endpoint without a token returns 401."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        r = await ac.get("/api/v1/auth/me")
    assert r.status_code == 401


async def test_invalid_token_returns_401():
    """Calling with an invalid JWT returns 401."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        r = await ac.get("/api/v1/auth/me", headers={"Authorization": "Bearer invalid.token.here"})
    assert r.status_code == 401


async def test_inactive_user_returns_401():
    """get_current_user raises UnauthorizedError for inactive users."""
    from app.core.dependencies import get_user_repository
    from app.domains.auth.repository import UserRepository
    from tests.conftest import make_user_dto

    inactive = make_user_dto(is_active=False)
    mock_repo = AsyncMock(spec=UserRepository)
    mock_repo.find_by_id = AsyncMock(return_value=inactive)

    app.dependency_overrides[get_user_repository] = lambda: mock_repo
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            token = __import__("app.core.security", fromlist=["create_access_token"]).create_access_token(inactive.id)
            r = await ac.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 401
    finally:
        app.dependency_overrides.clear()


async def test_user_not_found_returns_401():
    """get_current_user raises UnauthorizedError if user not found in DB."""
    from app.core.dependencies import get_user_repository
    from app.domains.auth.repository import UserRepository
    from app.core.security import create_access_token

    mock_repo = AsyncMock(spec=UserRepository)
    mock_repo.find_by_id = AsyncMock(return_value=None)

    app.dependency_overrides[get_user_repository] = lambda: mock_repo
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            token = create_access_token("deleted-user-id")
            r = await ac.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 401
    finally:
        app.dependency_overrides.clear()


# ── require_quota — QuotaExceededError → 429 ──────────────────────────────────

async def test_quota_exceeded_returns_429():
    """require_quota raises QuotaExceededError → 429 with feature/limit/used."""
    from app.core.dependencies import get_diagnosis_service

    mock_usage_svc = AsyncMock()
    mock_usage_svc.check_quota = AsyncMock(
        side_effect=QuotaExceededError(FeatureTypeEnum.INFERENCE, limit=5, used=5)
    )
    mock_usage_svc.record_usage = AsyncMock()

    app.dependency_overrides[get_usage_service] = lambda: mock_usage_svc
    app.dependency_overrides[get_current_user] = lambda: make_user_dto()
    app.dependency_overrides[get_diagnosis_service] = lambda: AsyncMock()
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            r = await ac.post(
                "/api/v1/diagnoses",
                json={
                    "disease_name": "X",
                    "disease_id": "x",
                    "confidence": 0.9,
                    "severity": "alta",
                    "model_used": "ensemble",
                    "top3": [],
                },
            )
        assert r.status_code == 429
        body = r.json()
        assert body["feature"] == "inference"
        assert body["limit"] == 5
        assert body["used"] == 5
    finally:
        app.dependency_overrides.clear()
