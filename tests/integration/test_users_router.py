"""Integration tests for /api/v1/users router."""

from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.dependencies import get_current_user, get_user_service
from app.core.exceptions import ConflictError, NotFoundError
from app.main import app
from tests.conftest import make_user_dto
from app.domains.users.schemas import UserProfileResponse
from tests.conftest import NOW


def make_profile_response(**kwargs) -> UserProfileResponse:
    user = make_user_dto(**kwargs)
    return UserProfileResponse(
        id=user.id,
        email=user.email,
        full_name=user.full_name,
        is_active=user.is_active,
        created_at=user.created_at,
        updated_at=user.updated_at,
    )


@pytest.fixture
def mock_user_svc():
    svc = AsyncMock()
    svc.get_profile = AsyncMock(return_value=make_profile_response())
    svc.update_profile = AsyncMock(return_value=make_profile_response())
    svc.delete_account = AsyncMock()
    return svc


@pytest.fixture
async def client_users(mock_user_svc):
    app.dependency_overrides[get_user_service] = lambda: mock_user_svc
    app.dependency_overrides[get_current_user] = lambda: make_user_dto()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()


# ── GET /users/me ─────────────────────────────────────────────────────────────

async def test_get_me_200(client_users):
    r = await client_users.get("/api/v1/users/me")
    assert r.status_code == 200
    assert r.json()["email"] == "test@example.com"


async def test_get_me_not_found(client_users, mock_user_svc):
    mock_user_svc.get_profile.side_effect = NotFoundError("User")
    r = await client_users.get("/api/v1/users/me")
    assert r.status_code == 404


# ── PATCH /users/me ───────────────────────────────────────────────────────────

async def test_patch_me_200(client_users):
    r = await client_users.patch("/api/v1/users/me", json={"full_name": "New Name"})
    assert r.status_code == 200


async def test_patch_me_conflict(client_users, mock_user_svc):
    mock_user_svc.update_profile.side_effect = ConflictError("Email already in use")
    r = await client_users.patch("/api/v1/users/me", json={"email": "taken@test.com"})
    assert r.status_code == 409


# ── DELETE /users/me ──────────────────────────────────────────────────────────

async def test_delete_me_204(client_users):
    r = await client_users.delete("/api/v1/users/me")
    assert r.status_code == 204
