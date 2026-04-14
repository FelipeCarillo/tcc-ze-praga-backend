"""Shared fixtures for integration tests."""

from datetime import datetime
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.dependencies import (
    get_action_plan_service,
    get_auth_service,
    get_current_user,
    get_diagnosis_service,
    get_subscription_service,
    get_usage_service,
    get_user_service,
)
from app.main import app
from tests.conftest import (
    make_action_plan_dto,
    make_diagnosis_dto,
    make_plan_dto,
    make_subscription_dto,
    make_usage_log_dto,
    make_user_dto,
    NOW,
)
from app.domains.auth.schemas import TokenResponse, UserResponse
from app.domains.diagnoses.schemas import DiagnosisResponse, Top3PredictionSchema
from app.domains.subscriptions.schemas import PlanResponse, SubscriptionResponse
from app.domains.usage.schemas import FeatureUsageResponse, UsageSummaryResponse, UsageHistoryItemResponse
from app.shared.enums import FeatureTypeEnum


# ── Shared response fixtures ──────────────────────────────────────────────────

def make_user_response() -> UserResponse:
    user = make_user_dto()
    return UserResponse(id=user.id, email=user.email, full_name=user.full_name, created_at=user.created_at)


def make_token_response() -> TokenResponse:
    return TokenResponse(access_token="fake-token", user=make_user_response())


def make_plan_response() -> PlanResponse:
    p = make_plan_dto()
    return PlanResponse(
        id=p.id,
        name=p.name,
        display_name=p.display_name,
        chat_daily_limit=p.chat_daily_limit,
        inference_daily_limit=p.inference_daily_limit,
        api_monthly_limit=p.api_monthly_limit,
    )


def make_subscription_response() -> SubscriptionResponse:
    sub = make_subscription_dto()
    return SubscriptionResponse(
        id=sub.id,
        plan=make_plan_response(),
        started_at=sub.started_at,
        expires_at=sub.expires_at,
        is_active=sub.is_active,
    )


def make_diagnosis_response() -> DiagnosisResponse:
    d = make_diagnosis_dto()
    return DiagnosisResponse(
        id=d.id,
        disease_name=d.disease_name,
        disease_id=d.disease_id,
        scientific_name=d.scientific_name,
        confidence=d.confidence,
        severity=d.severity,
        description=d.description,
        model_used=d.model_used,
        image_url=d.image_url,
        image_name=d.image_name,
        created_at=d.created_at,
        top3=[
            Top3PredictionSchema(
                rank=t.rank,
                disease_name=t.disease_name,
                disease_id=t.disease_id,
                scientific_name=t.scientific_name,
                confidence=t.confidence,
                severity=t.severity,
            )
            for t in d.top3
        ],
    )


def make_usage_summary() -> UsageSummaryResponse:
    return UsageSummaryResponse(
        chat=FeatureUsageResponse(used=2, limit=10, remaining=8),
        inference=FeatureUsageResponse(used=1, limit=5, remaining=4),
        api=FeatureUsageResponse(used=0, limit=0, remaining=0),
    )


# ── Client factory ────────────────────────────────────────────────────────────

@pytest.fixture
async def client():
    """Clean client with no dependency overrides."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()


@pytest.fixture
def auth_headers(valid_token) -> dict:
    return {"Authorization": f"Bearer {valid_token}"}
