"""Tests for app/domains/auth/service.py — AuthService."""

from unittest.mock import AsyncMock, patch

import pytest

from app.core.exceptions import ConflictError, UnauthorizedError
from app.domains.auth.schemas import LoginRequest, RegisterRequest
from app.domains.auth.service import AuthService
from tests.conftest import make_user_dto


@pytest.fixture
def user_repo():
    repo = AsyncMock()
    repo.find_by_email = AsyncMock(return_value=None)
    repo.create = AsyncMock(return_value=make_user_dto())
    return repo


# ── register ──────────────────────────────────────────────────────────────────

async def test_register_success(user_repo):
    with patch("app.domains.auth.service.hash_password", return_value="hashed"):
        with patch("app.domains.auth.service.create_access_token", return_value="tok"):
            svc = AuthService(user_repo)
            resp = await svc.register(
                RegisterRequest(email="new@test.com", password="secret1")
            )
    assert resp.access_token == "tok"
    assert resp.user.email == "test@example.com"
    user_repo.create.assert_awaited_once()


async def test_register_conflict(user_repo):
    user_repo.find_by_email.return_value = make_user_dto()
    svc = AuthService(user_repo)
    with pytest.raises(ConflictError, match="Email already in use"):
        await svc.register(RegisterRequest(email="test@example.com", password="secret1"))


# ── login ─────────────────────────────────────────────────────────────────────

async def test_login_success(user_repo):
    user_repo.find_by_email.return_value = make_user_dto(password_hash="hashed")
    with patch("app.domains.auth.service.verify_password", return_value=True):
        with patch("app.domains.auth.service.create_access_token", return_value="tok"):
            svc = AuthService(user_repo)
            resp = await svc.login(LoginRequest(email="test@example.com", password="pass"))
    assert resp.access_token == "tok"


async def test_login_user_not_found(user_repo):
    user_repo.find_by_email.return_value = None
    svc = AuthService(user_repo)
    with pytest.raises(UnauthorizedError, match="Invalid email or password"):
        await svc.login(LoginRequest(email="ghost@test.com", password="pass"))


async def test_login_wrong_password(user_repo):
    user_repo.find_by_email.return_value = make_user_dto()
    with patch("app.domains.auth.service.verify_password", return_value=False):
        svc = AuthService(user_repo)
        with pytest.raises(UnauthorizedError, match="Invalid email or password"):
            await svc.login(LoginRequest(email="test@example.com", password="wrong"))


async def test_login_inactive_user(user_repo):
    user_repo.find_by_email.return_value = make_user_dto(is_active=False)
    with patch("app.domains.auth.service.verify_password", return_value=True):
        svc = AuthService(user_repo)
        with pytest.raises(UnauthorizedError, match="Account is inactive"):
            await svc.login(LoginRequest(email="test@example.com", password="pass"))


# ── _build_token_response ─────────────────────────────────────────────────────

def test_build_token_response():
    user = make_user_dto()
    with patch("app.domains.auth.service.create_access_token", return_value="mytoken"):
        resp = AuthService._build_token_response(user)
    assert resp.access_token == "mytoken"
    assert resp.token_type == "bearer"
    assert resp.user.id == user.id
