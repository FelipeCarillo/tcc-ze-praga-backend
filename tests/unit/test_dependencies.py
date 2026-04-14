"""Tests for app/core/dependencies.py — factory functions and get_db."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from tests.conftest import make_user_dto


# ── get_db ────────────────────────────────────────────────────────────────────

async def test_get_db_yields_session():
    mock_session = AsyncMock(spec=AsyncSession)
    mock_ctx = MagicMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
    mock_ctx.__aexit__ = AsyncMock(return_value=None)

    with patch("app.db.database.AsyncSessionLocal", return_value=mock_ctx):
        from app.db.database import get_db

        gen = get_db()
        session = await gen.__anext__()
        assert session is mock_session
        try:
            await gen.__anext__()
        except StopAsyncIteration:
            pass


# ── Repository factory functions ──────────────────────────────────────────────

def test_get_subscription_repository():
    from app.core.dependencies import get_subscription_repository
    from app.domains.subscriptions.repository import SubscriptionRepository

    db = MagicMock(spec=AsyncSession)
    result = get_subscription_repository(db)
    assert isinstance(result, SubscriptionRepository)


def test_get_usage_repository():
    from app.core.dependencies import get_usage_repository
    from app.domains.usage.repository import UsageRepository

    db = MagicMock(spec=AsyncSession)
    result = get_usage_repository(db)
    assert isinstance(result, UsageRepository)


def test_get_diagnosis_repository():
    from app.core.dependencies import get_diagnosis_repository
    from app.domains.diagnoses.repository import DiagnosisRepository

    db = MagicMock(spec=AsyncSession)
    result = get_diagnosis_repository(db)
    assert isinstance(result, DiagnosisRepository)


def test_get_action_plan_repository():
    from app.core.dependencies import get_action_plan_repository
    from app.domains.action_plans.repository import ActionPlanRepository

    db = MagicMock(spec=AsyncSession)
    result = get_action_plan_repository(db)
    assert isinstance(result, ActionPlanRepository)


# ── Service factory functions ─────────────────────────────────────────────────

def test_get_auth_service():
    from app.core.dependencies import get_auth_service
    from app.domains.auth.repository import UserRepository
    from app.domains.auth.service import AuthService

    repo = MagicMock(spec=UserRepository)
    result = get_auth_service(repo)
    assert isinstance(result, AuthService)


def test_get_user_service():
    from app.core.dependencies import get_user_service
    from app.domains.auth.repository import UserRepository
    from app.domains.users.service import UserService

    repo = MagicMock(spec=UserRepository)
    result = get_user_service(repo)
    assert isinstance(result, UserService)


def test_get_subscription_service():
    from app.core.dependencies import get_subscription_service
    from app.domains.subscriptions.repository import SubscriptionRepository
    from app.domains.subscriptions.service import SubscriptionService

    repo = MagicMock(spec=SubscriptionRepository)
    result = get_subscription_service(repo)
    assert isinstance(result, SubscriptionService)


def test_get_usage_service():
    from app.core.dependencies import get_usage_service
    from app.domains.subscriptions.repository import SubscriptionRepository
    from app.domains.usage.repository import UsageRepository
    from app.domains.usage.service import UsageService

    usage_repo = MagicMock(spec=UsageRepository)
    sub_repo = MagicMock(spec=SubscriptionRepository)
    result = get_usage_service(usage_repo, sub_repo)
    assert isinstance(result, UsageService)


def test_get_diagnosis_service():
    from app.core.dependencies import get_diagnosis_service
    from app.domains.diagnoses.repository import DiagnosisRepository
    from app.domains.diagnoses.service import DiagnosisService

    repo = MagicMock(spec=DiagnosisRepository)
    result = get_diagnosis_service(repo)
    assert isinstance(result, DiagnosisService)


def test_get_action_plan_service():
    from app.core.dependencies import get_action_plan_service
    from app.domains.action_plans.repository import ActionPlanRepository
    from app.domains.action_plans.service import ActionPlanService

    repo = MagicMock(spec=ActionPlanRepository)
    result = get_action_plan_service(repo)
    assert isinstance(result, ActionPlanService)


# ── get_current_user — success path ──────────────────────────────────────────

async def test_get_current_user_success():
    """Covers the happy path return statement (line 99)."""
    from app.core.dependencies import get_current_user
    from app.core.security import create_access_token
    from app.domains.auth.repository import UserRepository

    user = make_user_dto()
    mock_repo = AsyncMock(spec=UserRepository)
    mock_repo.find_by_id = AsyncMock(return_value=user)

    token = create_access_token(user.id)
    result = await get_current_user(token=token, repo=mock_repo)
    assert result.id == user.id
