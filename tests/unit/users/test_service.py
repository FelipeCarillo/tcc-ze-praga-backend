"""Tests for app/domains/users/service.py — UserService."""

from unittest.mock import AsyncMock

import pytest

from app.core.exceptions import ConflictError, NotFoundError
from app.domains.users.schemas import UpdateUserRequest
from app.domains.users.service import UserService
from tests.conftest import make_plan_dto, make_subscription_dto, make_user_dto


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


# ── plan resolution (TCC-049) ────────────────────────────────────────────────


async def test_get_profile_without_sub_repo_has_no_plan(user_repo):
    """Quando sub_repo nao eh injetado, profile.plan deve ser None."""
    svc = UserService(user_repo, sub_repo=None)
    result = await svc.get_profile("user-uuid-1")
    assert result.plan is None


async def test_get_profile_without_active_subscription_has_no_plan(user_repo):
    """Usuario sem subscription ativa -> plan=None."""
    sub_repo = AsyncMock()
    sub_repo.find_user_subscription = AsyncMock(return_value=None)
    svc = UserService(user_repo, sub_repo=sub_repo)
    result = await svc.get_profile("user-uuid-1")
    assert result.plan is None


async def test_get_profile_with_subscription_includes_plan_features(user_repo):
    """Profile retorna plan.features quando subscription ativa existe."""
    features_dict = {"tier_name": "pro", "llm_model": "gpt-4o", "search_web": True}
    plan = make_plan_dto(name="pro", features=features_dict)
    sub = make_subscription_dto(plan=plan)
    sub_repo = AsyncMock()
    sub_repo.find_user_subscription = AsyncMock(return_value=sub)

    svc = UserService(user_repo, sub_repo=sub_repo)
    result = await svc.get_profile("user-uuid-1")
    assert result.plan is not None
    assert result.plan.name == "pro"
    assert result.plan.features == features_dict


async def test_update_profile_includes_plan_features(user_repo):
    features_dict = {"tier_name": "free", "llm_model": "gpt-4o-mini"}
    plan = make_plan_dto(features=features_dict)
    sub = make_subscription_dto(plan=plan)
    sub_repo = AsyncMock()
    sub_repo.find_user_subscription = AsyncMock(return_value=sub)

    svc = UserService(user_repo, sub_repo=sub_repo)
    result = await svc.update_profile(
        "user-uuid-1", UpdateUserRequest(full_name="Updated Name")
    )
    assert result.plan is not None
    assert result.plan.features == features_dict
