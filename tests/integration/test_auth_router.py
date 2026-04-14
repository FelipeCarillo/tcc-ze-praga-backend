"""Integration tests for /api/v1/auth router."""

from unittest.mock import AsyncMock

import pytest
from app.core.dependencies import get_auth_service, get_current_user
from app.core.exceptions import ConflictError, UnauthorizedError
from app.main import app
from tests.conftest import make_user_dto
from tests.integration.conftest import make_token_response, make_user_response


@pytest.fixture
def mock_auth_svc():
    svc = AsyncMock()
    svc.register = AsyncMock(return_value=make_token_response())
    svc.login = AsyncMock(return_value=make_token_response())
    return svc


@pytest.fixture
async def client_auth(mock_auth_svc):
    from httpx import ASGITransport, AsyncClient

    app.dependency_overrides[get_auth_service] = lambda: mock_auth_svc
    app.dependency_overrides[get_current_user] = lambda: make_user_dto()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()


# ── POST /auth/register ───────────────────────────────────────────────────────

async def test_register_201(client_auth):
    r = await client_auth.post(
        "/api/v1/auth/register",
        json={"email": "new@test.com", "password": "secret123"},
    )
    assert r.status_code == 201
    assert r.json()["access_token"] == "fake-token"


async def test_register_conflict_409(client_auth, mock_auth_svc):
    mock_auth_svc.register.side_effect = ConflictError("Email already in use")
    r = await client_auth.post(
        "/api/v1/auth/register",
        json={"email": "used@test.com", "password": "secret123"},
    )
    assert r.status_code == 409


async def test_register_invalid_body(client_auth):
    r = await client_auth.post("/api/v1/auth/register", json={"email": "bad"})
    assert r.status_code == 422


# ── POST /auth/login ──────────────────────────────────────────────────────────

async def test_login_200(client_auth):
    r = await client_auth.post(
        "/api/v1/auth/login",
        json={"email": "test@test.com", "password": "pass"},
    )
    assert r.status_code == 200
    assert "access_token" in r.json()


async def test_login_unauthorized_401(client_auth, mock_auth_svc):
    mock_auth_svc.login.side_effect = UnauthorizedError("Invalid email or password")
    r = await client_auth.post(
        "/api/v1/auth/login",
        json={"email": "x@x.com", "password": "wrong"},
    )
    assert r.status_code == 401


# ── GET /auth/me ──────────────────────────────────────────────────────────────

async def test_me_200(client_auth):
    r = await client_auth.get("/api/v1/auth/me")
    assert r.status_code == 200
    assert r.json()["email"] == "test@example.com"
