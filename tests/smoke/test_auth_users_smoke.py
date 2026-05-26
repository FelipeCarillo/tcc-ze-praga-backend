"""Smoke — auth + api-keys + users (TCC-069).

Camada fina: status do happy-path + 401/403 quando falta auth/tier + 1 campo-chave.
Mocks de service via dependency_overrides; gating Enterprise espelha
`tests/integration/test_api_keys_router.py`. Não duplica as asserts profundas da
integração.
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.dependencies import (
    get_api_key_service,
    get_auth_service,
    get_current_user,
    get_subscription_repository,
    get_user_service,
)
from app.domains.auth.api_key_dto import ApiKeyDTO
from app.domains.auth.schemas import TokenResponse, UserResponse
from app.domains.users.schemas import UserProfileResponse
from app.main import app
from tests.conftest import make_plan_dto, make_subscription_dto, make_user_dto
from tests.smoke.conftest import bypass_auth_overrides

NOW = datetime(2026, 5, 26, 12, 0, tzinfo=UTC)


def _token_response() -> TokenResponse:
    u = make_user_dto()
    return TokenResponse(
        access_token="fake-jwt",
        user=UserResponse(id=u.id, email=u.email, full_name=u.full_name, created_at=u.created_at),
    )


def _profile() -> UserProfileResponse:
    u = make_user_dto()
    return UserProfileResponse(
        id=u.id,
        email=u.email,
        full_name=u.full_name,
        is_active=True,
        created_at=NOW,
        updated_at=NOW,
        plan=None,
    )


def _api_key_dto() -> ApiKeyDTO:
    return ApiKeyDTO(
        id="apik-1",
        user_id="user-uuid-1",
        name="ci-key",
        key_hash="$2b$12$hash",
        key_prefix="zp_live_abcd",
        scopes=["diagnoses:analyze"],
        is_active=True,
        last_used_at=None,
        created_at=NOW,
        revoked_at=None,
    )


def _enterprise_sub_repo() -> AsyncMock:
    plan = make_plan_dto(
        id="plan-enterprise",
        name="enterprise",
        display_name="Enterprise",
        api_monthly_limit=10_000,
        features={"api_access": True},
    )
    repo = AsyncMock()
    repo.find_user_subscription = AsyncMock(return_value=make_subscription_dto(plan=plan))
    return repo


def _free_sub_repo() -> AsyncMock:
    repo = AsyncMock()
    repo.find_user_subscription = AsyncMock(
        return_value=make_subscription_dto(plan=make_plan_dto(features={"api_access": False}))
    )
    return repo


# ── Fixtures-client ─────────────────────────────────────────────────────────────


@pytest.fixture
async def client_auth():
    """Cliente com AuthService mockado (rotas públicas register/login + /auth/me)."""
    svc = AsyncMock()
    svc.register = AsyncMock(return_value=_token_response())
    svc.login = AsyncMock(return_value=_token_response())
    app.dependency_overrides[get_auth_service] = lambda: svc
    bypass_auth_overrides()  # libera GET /auth/me
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()


@pytest.fixture
async def client_users():
    svc = AsyncMock()
    svc.get_profile = AsyncMock(return_value=_profile())
    svc.update_profile = AsyncMock(return_value=_profile())
    svc.delete_account = AsyncMock(return_value=None)
    app.dependency_overrides[get_user_service] = lambda: svc
    bypass_auth_overrides()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()


def _api_key_svc() -> AsyncMock:
    svc = AsyncMock()
    svc.create = AsyncMock(return_value=(_api_key_dto(), "zp_live_abcd_plaintoken_xxxx"))
    svc.list_for_user = AsyncMock(return_value=[_api_key_dto()])
    svc.revoke = AsyncMock(return_value=True)
    return svc


async def _client_api_keys(sub_repo: AsyncMock) -> AsyncClient:
    app.dependency_overrides[get_current_user] = lambda: make_user_dto()
    app.dependency_overrides[get_api_key_service] = lambda: _api_key_svc()
    app.dependency_overrides[get_subscription_repository] = lambda: sub_repo
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


# ── auth (público + /me) ─────────────────────────────────────────────────────────


async def test_register_returns_201(client_auth):
    r = await client_auth.post(
        "/api/v1/auth/register",
        json={"email": "novo@example.com", "password": "secret123", "full_name": "Novo"},
    )
    assert r.status_code == 201
    assert r.json()["access_token"]


async def test_login_returns_200(client_auth):
    r = await client_auth.post(
        "/api/v1/auth/login", json={"email": "test@example.com", "password": "secret123"}
    )
    assert r.status_code == 200
    assert r.json()["user"]["email"]


async def test_me_returns_200(client_auth):
    r = await client_auth.get("/api/v1/auth/me")
    assert r.status_code == 200
    assert r.json()["id"]


async def test_me_requires_auth(smoke_client):
    r = await smoke_client.get("/api/v1/auth/me")
    assert r.status_code == 401


# ── api-keys (gating Enterprise) ──────────────────────────────────────────────────


async def test_api_key_create_enterprise_201():
    client = await _client_api_keys(_enterprise_sub_repo())
    async with client as ac:
        r = await ac.post("/api/v1/auth/api-keys", json={"name": "ci-key"})
    app.dependency_overrides.clear()
    assert r.status_code == 201
    assert r.json()["key"].startswith("zp_live_")


async def test_api_key_list_enterprise_200():
    client = await _client_api_keys(_enterprise_sub_repo())
    async with client as ac:
        r = await ac.get("/api/v1/auth/api-keys")
    app.dependency_overrides.clear()
    assert r.status_code == 200
    assert r.json()[0]["key_prefix"] == "zp_live_abcd"


async def test_api_key_revoke_enterprise_204():
    client = await _client_api_keys(_enterprise_sub_repo())
    async with client as ac:
        r = await ac.delete("/api/v1/auth/api-keys/apik-1")
    app.dependency_overrides.clear()
    assert r.status_code == 204


async def test_api_key_forbidden_for_free_tier():
    client = await _client_api_keys(_free_sub_repo())
    async with client as ac:
        r = await ac.post("/api/v1/auth/api-keys", json={"name": "x"})
    app.dependency_overrides.clear()
    assert r.status_code == 403


# ── users/me ──────────────────────────────────────────────────────────────────────


async def test_users_get_me_200(client_users):
    r = await client_users.get("/api/v1/users/me")
    assert r.status_code == 200
    assert r.json()["id"]


async def test_users_patch_me_200(client_users):
    r = await client_users.patch("/api/v1/users/me", json={"full_name": "Atualizado"})
    assert r.status_code == 200


async def test_users_delete_me_204(client_users):
    r = await client_users.delete("/api/v1/users/me")
    assert r.status_code == 204


async def test_users_me_requires_auth(smoke_client):
    r = await smoke_client.get("/api/v1/users/me")
    assert r.status_code == 401
