"""Integration tests for /api/v1/auth/api-keys router (Enterprise tier only)."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.dependencies import (
    get_api_key_service,
    get_current_user,
    get_subscription_repository,
)
from app.domains.auth.api_key_dto import ApiKeyDTO
from app.main import app
from tests.conftest import make_plan_dto, make_subscription_dto, make_user_dto


NOW = datetime(2026, 5, 23, 12, 0, tzinfo=UTC)


def _dto(**kwargs) -> ApiKeyDTO:
    defaults = dict(
        id="apik-1",
        user_id="user-uuid-1",
        name="my-key",
        key_hash="$2b$12$hash",
        key_prefix="zp_live_abcd",
        scopes=["diagnoses:analyze"],
        is_active=True,
        last_used_at=None,
        created_at=NOW,
        revoked_at=None,
    )
    return ApiKeyDTO(**{**defaults, **kwargs})


def _enterprise_sub():
    plan = make_plan_dto(
        id="plan-enterprise",
        name="enterprise",
        display_name="Enterprise",
        chat_daily_limit=None,
        inference_daily_limit=None,
        api_monthly_limit=10_000,
        features={"api_access": True, "tier_name": "enterprise"},
    )
    return make_subscription_dto(plan=plan)


def _free_sub():
    plan = make_plan_dto(features={"api_access": False, "tier_name": "free"})
    return make_subscription_dto(plan=plan)


@pytest.fixture
def mock_api_key_svc():
    svc = AsyncMock()
    svc.create = AsyncMock(return_value=(_dto(), "zp_live_abcdEFGH_realtoken_xxxxxxxx"))
    svc.list_for_user = AsyncMock(return_value=[_dto()])
    svc.revoke = AsyncMock(return_value=True)
    return svc


@pytest.fixture
def mock_sub_repo_enterprise():
    repo = AsyncMock()
    repo.find_user_subscription = AsyncMock(return_value=_enterprise_sub())
    return repo


@pytest.fixture
def mock_sub_repo_free():
    repo = AsyncMock()
    repo.find_user_subscription = AsyncMock(return_value=_free_sub())
    return repo


@pytest.fixture
def mock_sub_repo_none():
    repo = AsyncMock()
    repo.find_user_subscription = AsyncMock(return_value=None)
    return repo


# ── helpers ───────────────────────────────────────────────────────────────────

async def _client_with_overrides(mock_api_key_svc, mock_sub_repo):
    app.dependency_overrides[get_current_user] = lambda: make_user_dto()
    app.dependency_overrides[get_api_key_service] = lambda: mock_api_key_svc
    app.dependency_overrides[get_subscription_repository] = lambda: mock_sub_repo
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


# ── POST /auth/api-keys ───────────────────────────────────────────────────────

async def test_create_enterprise_returns_plain_key_once(
    mock_api_key_svc, mock_sub_repo_enterprise
):
    client = await _client_with_overrides(mock_api_key_svc, mock_sub_repo_enterprise)
    async with client as ac:
        r = await ac.post("/api/v1/auth/api-keys", json={"name": "ci-key"})
    app.dependency_overrides.clear()

    assert r.status_code == 201
    body = r.json()
    assert body["id"] == "apik-1"
    assert body["key"].startswith("zp_live_")
    assert body["key_prefix"] == "zp_live_abcd"
    assert body["scopes"] == ["diagnoses:analyze"]


async def test_create_forbidden_for_free_tier(mock_api_key_svc, mock_sub_repo_free):
    client = await _client_with_overrides(mock_api_key_svc, mock_sub_repo_free)
    async with client as ac:
        r = await ac.post("/api/v1/auth/api-keys", json={"name": "x"})
    app.dependency_overrides.clear()

    assert r.status_code == 403


async def test_create_forbidden_when_no_subscription(
    mock_api_key_svc, mock_sub_repo_none
):
    client = await _client_with_overrides(mock_api_key_svc, mock_sub_repo_none)
    async with client as ac:
        r = await ac.post("/api/v1/auth/api-keys", json={"name": "x"})
    app.dependency_overrides.clear()

    assert r.status_code == 403


async def test_create_invalid_body_422(mock_api_key_svc, mock_sub_repo_enterprise):
    client = await _client_with_overrides(mock_api_key_svc, mock_sub_repo_enterprise)
    async with client as ac:
        r = await ac.post("/api/v1/auth/api-keys", json={"name": ""})
    app.dependency_overrides.clear()

    assert r.status_code == 422


# ── GET /auth/api-keys ────────────────────────────────────────────────────────

async def test_list_enterprise_returns_keys(mock_api_key_svc, mock_sub_repo_enterprise):
    client = await _client_with_overrides(mock_api_key_svc, mock_sub_repo_enterprise)
    async with client as ac:
        r = await ac.get("/api/v1/auth/api-keys")
    app.dependency_overrides.clear()

    assert r.status_code == 200
    body = r.json()
    assert isinstance(body, list)
    assert body[0]["id"] == "apik-1"
    assert body[0]["key_prefix"] == "zp_live_abcd"
    # nunca retorna o plain text na listagem
    assert "key" not in body[0]
    assert "key_hash" not in body[0]


async def test_list_forbidden_for_free_tier(mock_api_key_svc, mock_sub_repo_free):
    client = await _client_with_overrides(mock_api_key_svc, mock_sub_repo_free)
    async with client as ac:
        r = await ac.get("/api/v1/auth/api-keys")
    app.dependency_overrides.clear()

    assert r.status_code == 403


# ── DELETE /auth/api-keys/{id} ────────────────────────────────────────────────

async def test_revoke_returns_204(mock_api_key_svc, mock_sub_repo_enterprise):
    client = await _client_with_overrides(mock_api_key_svc, mock_sub_repo_enterprise)
    async with client as ac:
        r = await ac.delete("/api/v1/auth/api-keys/apik-1")
    app.dependency_overrides.clear()

    assert r.status_code == 204


async def test_revoke_404_when_not_found(mock_api_key_svc, mock_sub_repo_enterprise):
    mock_api_key_svc.revoke.return_value = False
    client = await _client_with_overrides(mock_api_key_svc, mock_sub_repo_enterprise)
    async with client as ac:
        r = await ac.delete("/api/v1/auth/api-keys/ghost")
    app.dependency_overrides.clear()

    assert r.status_code == 404


async def test_revoke_forbidden_for_free(mock_api_key_svc, mock_sub_repo_free):
    client = await _client_with_overrides(mock_api_key_svc, mock_sub_repo_free)
    async with client as ac:
        r = await ac.delete("/api/v1/auth/api-keys/apik-1")
    app.dependency_overrides.clear()

    assert r.status_code == 403
