"""Integration tests for X-RateLimit-* headers em POST /diagnoses/analyze (TCC-064).

So' devem aparecer quando autenticado via API key (monthly quota).
Quando via JWT (daily quota INFERENCE), headers nao sao retornados.
"""

import calendar
import io
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.dependencies import (
    auth_method_dual,
    get_current_user,
    get_current_user_or_api_key,
    get_diagnosis_graph_factory,
    get_diagnosis_repository,
    get_plan_features_dual,
    get_subscription_repository,
    get_usage_repository,
    get_usage_service,
    require_quota,
    require_quota_dual,
)
from app.domains.subscriptions.features import ENTERPRISE_FEATURES
from app.main import app
from app.shared.enums import FeatureTypeEnum
from tests.conftest import make_diagnosis_dto, make_plan_dto, make_subscription_dto, make_user_dto


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_graph_factory():
    async def _ainvoke(state):
        n = len(state.get("image_ids", []))
        return {
            "persisted_ids": [f"diag-{i}" for i in range(n)],
            "predictions": [{} for _ in range(n)],
        }

    graph = MagicMock()
    graph.ainvoke = _ainvoke

    def _factory_fn(crop_id):
        return graph

    return _factory_fn


@pytest.fixture
def mock_diag_repo():
    repo = AsyncMock()
    repo.find_by_id = AsyncMock(
        side_effect=lambda diag_id, _user_id: make_diagnosis_dto(id=diag_id)
    )
    return repo


@pytest.fixture
def mock_usage_svc():
    svc = AsyncMock()
    svc.check_quota = AsyncMock()
    svc.record_usage = AsyncMock()
    return svc


@pytest.fixture
def mock_usage_repo():
    repo = AsyncMock()
    repo.count_this_month = AsyncMock(return_value=42)
    return repo


@pytest.fixture
def mock_sub_repo_enterprise():
    repo = AsyncMock()
    enterprise_plan = make_plan_dto(
        id="plan-enterprise",
        name="enterprise",
        api_monthly_limit=500,
        features={"api_access": True, "tier_name": "enterprise"},
    )
    repo.find_user_subscription = AsyncMock(
        return_value=make_subscription_dto(plan=enterprise_plan)
    )
    return repo


async def _client_with_api_key(
    mock_graph_factory, mock_diag_repo, mock_usage_svc, mock_usage_repo, mock_sub_repo
):
    """Cliente com overrides simulando autenticacao via API key (auth_method='api_key')."""
    app.dependency_overrides[get_current_user_or_api_key] = lambda: make_user_dto()
    app.dependency_overrides[get_plan_features_dual] = lambda: ENTERPRISE_FEATURES
    app.dependency_overrides[auth_method_dual] = lambda: "api_key"
    app.dependency_overrides[require_quota_dual] = lambda: make_user_dto()
    # Backward-compat
    app.dependency_overrides[require_quota(FeatureTypeEnum.INFERENCE)] = (
        lambda: make_user_dto()
    )
    app.dependency_overrides[get_current_user] = lambda: make_user_dto()
    app.dependency_overrides[get_diagnosis_graph_factory] = lambda: mock_graph_factory
    app.dependency_overrides[get_diagnosis_repository] = lambda: mock_diag_repo
    app.dependency_overrides[get_usage_service] = lambda: mock_usage_svc
    app.dependency_overrides[get_usage_repository] = lambda: mock_usage_repo
    app.dependency_overrides[get_subscription_repository] = lambda: mock_sub_repo
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def _client_with_jwt(
    mock_graph_factory, mock_diag_repo, mock_usage_svc
):
    """Cliente com overrides simulando JWT (auth_method='jwt'). Sem rate-limit headers."""
    app.dependency_overrides[get_current_user_or_api_key] = lambda: make_user_dto()
    app.dependency_overrides[get_plan_features_dual] = lambda: ENTERPRISE_FEATURES
    app.dependency_overrides[auth_method_dual] = lambda: "jwt"
    app.dependency_overrides[require_quota_dual] = lambda: make_user_dto()
    app.dependency_overrides[require_quota(FeatureTypeEnum.INFERENCE)] = (
        lambda: make_user_dto()
    )
    app.dependency_overrides[get_current_user] = lambda: make_user_dto()
    app.dependency_overrides[get_diagnosis_graph_factory] = lambda: mock_graph_factory
    app.dependency_overrides[get_diagnosis_repository] = lambda: mock_diag_repo
    app.dependency_overrides[get_usage_service] = lambda: mock_usage_svc
    app.dependency_overrides[get_usage_repository] = lambda: AsyncMock()
    app.dependency_overrides[get_subscription_repository] = lambda: AsyncMock()
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


# ── Tests ─────────────────────────────────────────────────────────────────────


async def test_analyze_with_api_key_returns_rate_limit_headers(
    mock_graph_factory,
    mock_diag_repo,
    mock_usage_svc,
    mock_usage_repo,
    mock_sub_repo_enterprise,
):
    client = await _client_with_api_key(
        mock_graph_factory,
        mock_diag_repo,
        mock_usage_svc,
        mock_usage_repo,
        mock_sub_repo_enterprise,
    )
    async with client as ac:
        files = {"images": ("leaf.jpg", io.BytesIO(b"x"), "image/jpeg")}
        r = await ac.post("/api/v1/diagnoses/analyze", files=files)
    app.dependency_overrides.clear()

    assert r.status_code == 200
    assert "X-RateLimit-Limit" in r.headers
    assert "X-RateLimit-Remaining" in r.headers
    assert "X-RateLimit-Reset" in r.headers


async def test_rate_limit_headers_have_correct_values(
    mock_graph_factory,
    mock_diag_repo,
    mock_usage_svc,
    mock_usage_repo,
    mock_sub_repo_enterprise,
):
    """Plan limit=500, used=42 -> Limit=500, Remaining=458."""
    client = await _client_with_api_key(
        mock_graph_factory,
        mock_diag_repo,
        mock_usage_svc,
        mock_usage_repo,
        mock_sub_repo_enterprise,
    )
    async with client as ac:
        files = {"images": ("leaf.jpg", io.BytesIO(b"x"), "image/jpeg")}
        r = await ac.post("/api/v1/diagnoses/analyze", files=files)
    app.dependency_overrides.clear()

    assert r.headers["X-RateLimit-Limit"] == "500"
    assert r.headers["X-RateLimit-Remaining"] == "458"
    # Reset eh epoch — deve ser inteiro futuro
    reset_epoch = int(r.headers["X-RateLimit-Reset"])
    assert reset_epoch > int(datetime.now(UTC).timestamp())


async def test_rate_limit_reset_is_next_month_start(
    mock_graph_factory,
    mock_diag_repo,
    mock_usage_svc,
    mock_usage_repo,
    mock_sub_repo_enterprise,
):
    """Reset deve ser epoch UTC do inicio do proximo mes."""
    client = await _client_with_api_key(
        mock_graph_factory,
        mock_diag_repo,
        mock_usage_svc,
        mock_usage_repo,
        mock_sub_repo_enterprise,
    )
    async with client as ac:
        files = {"images": ("leaf.jpg", io.BytesIO(b"x"), "image/jpeg")}
        r = await ac.post("/api/v1/diagnoses/analyze", files=files)
    app.dependency_overrides.clear()

    reset_epoch = int(r.headers["X-RateLimit-Reset"])
    reset_dt = datetime.fromtimestamp(reset_epoch, tz=UTC)

    now = datetime.now(UTC)
    # Inicio do mes seguinte: day=1, sem horario depois (deveria ser perto de
    # 00:00:00 do dia 1 do proximo mes).
    if now.month == 12:
        expected_month, expected_year = 1, now.year + 1
    else:
        expected_month, expected_year = now.month + 1, now.year

    assert reset_dt.month == expected_month
    assert reset_dt.year == expected_year
    assert reset_dt.day == 1


async def test_rate_limit_remaining_zero_when_overused(
    mock_graph_factory,
    mock_diag_repo,
    mock_usage_svc,
    mock_usage_repo,
    mock_sub_repo_enterprise,
):
    """Remaining nao pode ser negativo — clamp em 0."""
    mock_usage_repo.count_this_month = AsyncMock(return_value=999)
    client = await _client_with_api_key(
        mock_graph_factory,
        mock_diag_repo,
        mock_usage_svc,
        mock_usage_repo,
        mock_sub_repo_enterprise,
    )
    async with client as ac:
        files = {"images": ("leaf.jpg", io.BytesIO(b"x"), "image/jpeg")}
        r = await ac.post("/api/v1/diagnoses/analyze", files=files)
    app.dependency_overrides.clear()

    assert r.headers["X-RateLimit-Remaining"] == "0"


async def test_rate_limit_limit_zero_when_no_subscription(
    mock_graph_factory,
    mock_diag_repo,
    mock_usage_svc,
    mock_usage_repo,
):
    """User sem subscription -> limit=0, remaining=0."""
    sub_repo = AsyncMock()
    sub_repo.find_user_subscription = AsyncMock(return_value=None)

    client = await _client_with_api_key(
        mock_graph_factory, mock_diag_repo, mock_usage_svc, mock_usage_repo, sub_repo
    )
    async with client as ac:
        files = {"images": ("leaf.jpg", io.BytesIO(b"x"), "image/jpeg")}
        r = await ac.post("/api/v1/diagnoses/analyze", files=files)
    app.dependency_overrides.clear()

    assert r.headers["X-RateLimit-Limit"] == "0"
    assert r.headers["X-RateLimit-Remaining"] == "0"


async def test_analyze_with_jwt_does_not_emit_rate_limit_headers(
    mock_graph_factory, mock_diag_repo, mock_usage_svc
):
    """Auth via JWT -> sem headers (daily quota tem outro tipo)."""
    client = await _client_with_jwt(mock_graph_factory, mock_diag_repo, mock_usage_svc)
    async with client as ac:
        files = {"images": ("leaf.jpg", io.BytesIO(b"x"), "image/jpeg")}
        r = await ac.post("/api/v1/diagnoses/analyze", files=files)
    app.dependency_overrides.clear()

    assert r.status_code == 200
    assert "X-RateLimit-Limit" not in r.headers
    assert "X-RateLimit-Remaining" not in r.headers


async def test_api_key_records_usage_with_api_feature(
    mock_graph_factory,
    mock_diag_repo,
    mock_usage_svc,
    mock_usage_repo,
    mock_sub_repo_enterprise,
):
    """Quando autenticado via API key, usage_log usa feature=API (monthly)."""
    client = await _client_with_api_key(
        mock_graph_factory,
        mock_diag_repo,
        mock_usage_svc,
        mock_usage_repo,
        mock_sub_repo_enterprise,
    )
    async with client as ac:
        files = {"images": ("leaf.jpg", io.BytesIO(b"x"), "image/jpeg")}
        await ac.post("/api/v1/diagnoses/analyze", files=files)
    app.dependency_overrides.clear()

    # record_usage(user_id, feature, metadata) — feature deve ser API
    feature_arg = mock_usage_svc.record_usage.await_args.args[1]
    metadata = mock_usage_svc.record_usage.await_args.args[2]
    assert feature_arg == FeatureTypeEnum.API
    assert metadata["auth_method"] == "api_key"


async def test_jwt_records_usage_with_inference_feature(
    mock_graph_factory, mock_diag_repo, mock_usage_svc
):
    """Quando autenticado via JWT, usage_log usa feature=INFERENCE (daily)."""
    client = await _client_with_jwt(mock_graph_factory, mock_diag_repo, mock_usage_svc)
    async with client as ac:
        files = {"images": ("leaf.jpg", io.BytesIO(b"x"), "image/jpeg")}
        await ac.post("/api/v1/diagnoses/analyze", files=files)
    app.dependency_overrides.clear()

    feature_arg = mock_usage_svc.record_usage.await_args.args[1]
    metadata = mock_usage_svc.record_usage.await_args.args[2]
    assert feature_arg == FeatureTypeEnum.INFERENCE
    assert metadata["auth_method"] == "jwt"
