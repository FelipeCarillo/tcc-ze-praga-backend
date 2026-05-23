"""Unit tests for ``get_current_user_or_api_key``, ``auth_method_dual``,
``require_quota_dual``."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from app.core.dependencies import (
    auth_method_dual,
    get_current_user_or_api_key,
    require_quota_dual,
)
from app.core.exceptions import QuotaExceededError, UnauthorizedError
from app.core.security import create_access_token
from app.domains.auth.api_key_dto import ApiKeyDTO
from app.shared.enums import FeatureTypeEnum
from tests.conftest import make_user_dto

NOW = datetime(2026, 5, 23, tzinfo=UTC)


def _api_key_dto(**kwargs) -> ApiKeyDTO:
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


# ── auth_method_dual ──────────────────────────────────────────────────────────


async def test_auth_method_dual_api_key_present():
    out = await auth_method_dual(x_api_key="zp_live_xxxx")
    assert out == "api_key"


async def test_auth_method_dual_no_api_key():
    out = await auth_method_dual(x_api_key=None)
    assert out == "jwt"


# ── get_current_user_or_api_key — API key path ────────────────────────────────


async def test_api_key_path_returns_user():
    user_repo = AsyncMock()
    user_repo.find_by_id.return_value = make_user_dto()

    api_key_svc = AsyncMock()
    api_key_svc.verify.return_value = _api_key_dto()

    user = await get_current_user_or_api_key(
        authorization=None,
        x_api_key="zp_live_abcd_realtoken",
        user_repo=user_repo,
        api_key_svc=api_key_svc,
    )
    assert user.id == "user-uuid-1"
    api_key_svc.touch_last_used.assert_awaited_once_with("apik-1")


async def test_api_key_path_invalid_key_raises_401():
    user_repo = AsyncMock()
    api_key_svc = AsyncMock()
    api_key_svc.verify.return_value = None

    with pytest.raises(UnauthorizedError):
        await get_current_user_or_api_key(
            authorization=None,
            x_api_key="zp_live_garbage",
            user_repo=user_repo,
            api_key_svc=api_key_svc,
        )


async def test_api_key_path_inactive_key_raises_401():
    user_repo = AsyncMock()
    api_key_svc = AsyncMock()
    api_key_svc.verify.return_value = _api_key_dto(is_active=False)

    with pytest.raises(UnauthorizedError):
        await get_current_user_or_api_key(
            authorization=None,
            x_api_key="zp_live_xxxx",
            user_repo=user_repo,
            api_key_svc=api_key_svc,
        )


async def test_api_key_path_inactive_user_raises_401():
    user_repo = AsyncMock()
    user_repo.find_by_id.return_value = make_user_dto(is_active=False)

    api_key_svc = AsyncMock()
    api_key_svc.verify.return_value = _api_key_dto()

    with pytest.raises(UnauthorizedError):
        await get_current_user_or_api_key(
            authorization=None,
            x_api_key="zp_live_xxxx",
            user_repo=user_repo,
            api_key_svc=api_key_svc,
        )


async def test_api_key_takes_precedence_over_jwt():
    """Quando ambos headers presentes, X-API-Key ganha."""
    user_repo = AsyncMock()
    user_repo.find_by_id.return_value = make_user_dto()

    api_key_svc = AsyncMock()
    api_key_svc.verify.return_value = _api_key_dto()

    # Authorization JWT presente mas API key tambem — usa API
    user = await get_current_user_or_api_key(
        authorization="Bearer some-jwt-token",
        x_api_key="zp_live_xxxx",
        user_repo=user_repo,
        api_key_svc=api_key_svc,
    )
    assert user.id == "user-uuid-1"
    api_key_svc.verify.assert_awaited_once()
    # JWT nao foi decodificado (fluxo terminou no API path)


# ── get_current_user_or_api_key — JWT path ────────────────────────────────────


async def test_jwt_path_returns_user():
    user_repo = AsyncMock()
    user_repo.find_by_id.return_value = make_user_dto()

    api_key_svc = AsyncMock()

    token = create_access_token("user-uuid-1")
    user = await get_current_user_or_api_key(
        authorization=f"Bearer {token}",
        x_api_key=None,
        user_repo=user_repo,
        api_key_svc=api_key_svc,
    )
    assert user.id == "user-uuid-1"
    api_key_svc.verify.assert_not_called()


async def test_jwt_path_invalid_token_raises_401():
    user_repo = AsyncMock()
    api_key_svc = AsyncMock()

    with pytest.raises(UnauthorizedError):
        await get_current_user_or_api_key(
            authorization="Bearer garbage-token",
            x_api_key=None,
            user_repo=user_repo,
            api_key_svc=api_key_svc,
        )


async def test_jwt_path_inactive_user_raises_401():
    user_repo = AsyncMock()
    user_repo.find_by_id.return_value = make_user_dto(is_active=False)
    api_key_svc = AsyncMock()

    token = create_access_token("user-uuid-1")
    with pytest.raises(UnauthorizedError):
        await get_current_user_or_api_key(
            authorization=f"Bearer {token}",
            x_api_key=None,
            user_repo=user_repo,
            api_key_svc=api_key_svc,
        )


# ── neither header present ────────────────────────────────────────────────────


async def test_no_auth_raises_401():
    with pytest.raises(UnauthorizedError):
        await get_current_user_or_api_key(
            authorization=None,
            x_api_key=None,
            user_repo=AsyncMock(),
            api_key_svc=AsyncMock(),
        )


async def test_authorization_without_bearer_prefix_raises_401():
    with pytest.raises(UnauthorizedError):
        await get_current_user_or_api_key(
            authorization="Basic abc",
            x_api_key=None,
            user_repo=AsyncMock(),
            api_key_svc=AsyncMock(),
        )


# ── require_quota_dual ────────────────────────────────────────────────────────


async def test_require_quota_dual_api_key_checks_API_feature():
    usage_svc = AsyncMock()
    user = await require_quota_dual(
        current_user=make_user_dto(),
        auth_method="api_key",
        usage_svc=usage_svc,
    )
    assert user.id == "user-uuid-1"
    usage_svc.check_quota.assert_awaited_once_with("user-uuid-1", FeatureTypeEnum.API)


async def test_require_quota_dual_jwt_checks_INFERENCE_feature():
    usage_svc = AsyncMock()
    await require_quota_dual(
        current_user=make_user_dto(),
        auth_method="jwt",
        usage_svc=usage_svc,
    )
    usage_svc.check_quota.assert_awaited_once_with(
        "user-uuid-1", FeatureTypeEnum.INFERENCE
    )


async def test_require_quota_dual_propagates_quota_exceeded():
    usage_svc = AsyncMock()
    usage_svc.check_quota.side_effect = QuotaExceededError(
        FeatureTypeEnum.API, limit=100, used=100
    )
    with pytest.raises(QuotaExceededError):
        await require_quota_dual(
            current_user=make_user_dto(),
            auth_method="api_key",
            usage_svc=usage_svc,
        )
