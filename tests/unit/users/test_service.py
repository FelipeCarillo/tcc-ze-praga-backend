"""Tests for app/domains/users/service.py — UserService."""

from unittest.mock import AsyncMock

import pytest

from app.core.exceptions import ConflictError, NotFoundError
from app.domains.users.schemas import UpdateUserRequest
from app.domains.users.service import UserService
from tests.conftest import make_user_dto


@pytest.fixture
def user_repo():
    repo = AsyncMock()
    repo.find_by_id = AsyncMock(return_value=make_user_dto())
    repo.find_by_email = AsyncMock(return_value=None)
    repo.update = AsyncMock(return_value=make_user_dto())
    repo.soft_delete = AsyncMock()
    return repo


# ── get_profile ───────────────────────────────────────────────────────────────

async def test_get_profile_success(user_repo):
    svc = UserService(user_repo)
    result = await svc.get_profile("user-uuid-1")
    assert result.email == "test@example.com"


async def test_get_profile_not_found(user_repo):
    user_repo.find_by_id.return_value = None
    svc = UserService(user_repo)
    with pytest.raises(NotFoundError, match="User"):
        await svc.get_profile("ghost")


# ── update_profile ────────────────────────────────────────────────────────────

async def test_update_profile_no_email(user_repo):
    svc = UserService(user_repo)
    result = await svc.update_profile("user-uuid-1", UpdateUserRequest(full_name="New Name"))
    assert result is not None
    user_repo.find_by_email.assert_not_called()


async def test_update_profile_new_email_no_conflict(user_repo):
    user_repo.find_by_email.return_value = None
    svc = UserService(user_repo)
    result = await svc.update_profile(
        "user-uuid-1", UpdateUserRequest(email="new@test.com")
    )
    assert result is not None


async def test_update_profile_same_user_email(user_repo):
    """Changing to your own current email should not raise ConflictError."""
    user_repo.find_by_email.return_value = make_user_dto(id="user-uuid-1")
    svc = UserService(user_repo)
    result = await svc.update_profile(
        "user-uuid-1", UpdateUserRequest(email="test@example.com")
    )
    assert result is not None


async def test_update_profile_email_conflict(user_repo):
    """Email already used by a different user → ConflictError."""
    user_repo.find_by_email.return_value = make_user_dto(id="other-user")
    svc = UserService(user_repo)
    with pytest.raises(ConflictError, match="Email already in use"):
        await svc.update_profile(
            "user-uuid-1", UpdateUserRequest(email="other@test.com")
        )


async def test_update_profile_user_not_found_after_update(user_repo):
    user_repo.update.return_value = None
    svc = UserService(user_repo)
    with pytest.raises(NotFoundError, match="User"):
        await svc.update_profile("user-uuid-1", UpdateUserRequest(full_name="X"))


# ── delete_account ────────────────────────────────────────────────────────────

async def test_delete_account_calls_soft_delete(user_repo):
    svc = UserService(user_repo)
    await svc.delete_account("user-uuid-1")
    user_repo.soft_delete.assert_awaited_once_with("user-uuid-1")


# ── _to_profile ───────────────────────────────────────────────────────────────

def test_to_profile_mapping():
    user = make_user_dto(full_name=None)
    result = UserService._to_profile(user)
    assert result.id == user.id
    assert result.full_name is None
    assert result.is_active is True
