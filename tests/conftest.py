"""Global test fixtures shared across all test modules."""

import os

# Set required env vars BEFORE any app.* imports so Settings() succeeds without a .env file
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://test:test@localhost:5432/test")
os.environ.setdefault("SUPABASE_URL", "https://test.supabase.co")
os.environ.setdefault("SUPABASE_SERVICE_ROLE_KEY", "test-service-role-key")
os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-for-testing-only-32-chars!")

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.security import create_access_token
from app.domains.action_plans.dto import ActionPlanDTO, ActionPlanLevelDTO, ActionPlanSourceDTO
from app.domains.auth.dto import UserDTO
from app.domains.diagnoses.dto import DiagnosisDTO, Top3PredictionDTO
from app.domains.subscriptions.dto import PlanDTO, SubscriptionDTO
from app.domains.usage.dto import UsageLogDTO
from app.shared.enums import FeatureTypeEnum, SeverityEnum


# ── Helpers ───────────────────────────────────────────────────────────────────

NOW = datetime(2026, 4, 13, 12, 0, 0, tzinfo=UTC)


def make_user_dto(**kwargs) -> UserDTO:
    defaults = dict(
        id="user-uuid-1",
        email="test@example.com",
        password_hash="$2b$12$hashed",
        full_name="Test User",
        is_active=True,
        created_at=NOW,
        updated_at=NOW,
    )
    return UserDTO(**{**defaults, **kwargs})


def make_plan_dto(**kwargs) -> PlanDTO:
    defaults = dict(
        id="plan-uuid-1",
        name="free",
        display_name="Gratuito",
        chat_daily_limit=10,
        inference_daily_limit=5,
        api_monthly_limit=0,
        is_active=True,
    )
    return PlanDTO(**{**defaults, **kwargs})


def make_subscription_dto(plan: PlanDTO | None = None, **kwargs) -> SubscriptionDTO:
    plan = plan or make_plan_dto()
    defaults = dict(
        id="sub-uuid-1",
        user_id="user-uuid-1",
        plan=plan,
        started_at=NOW,
        expires_at=None,
        is_active=True,
    )
    return SubscriptionDTO(**{**defaults, **kwargs})


def make_diagnosis_dto(**kwargs) -> DiagnosisDTO:
    defaults = dict(
        id="diag-uuid-1",
        user_id="user-uuid-1",
        disease_name="Ferrugem Asiática",
        disease_id="ferrugem-asiatica",
        scientific_name="Phakopsora pachyrhizi",
        confidence=0.942,
        severity="alta",
        description="Descrição da doença",
        model_used="ensemble",
        image_url="http://example.com/img.jpg",
        image_name="image.jpg",
        created_at=NOW,
        top3=[
            Top3PredictionDTO(
                rank=1,
                disease_name="Ferrugem Asiática",
                disease_id="ferrugem-asiatica",
                scientific_name="Phakopsora pachyrhizi",
                confidence=0.942,
                severity="alta",
            )
        ],
    )
    return DiagnosisDTO(**{**defaults, **kwargs})


def make_action_plan_dto(**kwargs) -> ActionPlanDTO:
    defaults = dict(
        disease_id="ferrugem-asiatica",
        levels=[
            ActionPlanLevelDTO(
                disease_id="ferrugem-asiatica",
                level="essencial",
                actions=["Ação 1", "Ação 2"],
            )
        ],
        sources=[
            ActionPlanSourceDTO(
                id="src-uuid-1",
                name="EMBRAPA",
                detail="Fonte técnica",
                url="https://embrapa.br",
                display_order=0,
            )
        ],
    )
    return ActionPlanDTO(**{**defaults, **kwargs})


def make_usage_log_dto(**kwargs) -> UsageLogDTO:
    defaults = dict(
        id="log-uuid-1",
        feature=FeatureTypeEnum.INFERENCE,
        used_at=NOW,
        metadata=None,
    )
    return UsageLogDTO(**{**defaults, **kwargs})


# ── Pytest fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def fake_user() -> UserDTO:
    return make_user_dto()


@pytest.fixture
def fake_inactive_user() -> UserDTO:
    return make_user_dto(is_active=False)


@pytest.fixture
def fake_plan() -> PlanDTO:
    return make_plan_dto()


@pytest.fixture
def fake_pro_plan() -> PlanDTO:
    return make_plan_dto(
        id="plan-uuid-pro",
        name="pro",
        display_name="Pro",
        chat_daily_limit=None,
        inference_daily_limit=None,
        api_monthly_limit=500,
    )


@pytest.fixture
def fake_subscription(fake_plan) -> SubscriptionDTO:
    return make_subscription_dto(plan=fake_plan)


@pytest.fixture
def fake_diagnosis() -> DiagnosisDTO:
    return make_diagnosis_dto()


@pytest.fixture
def fake_action_plan() -> ActionPlanDTO:
    return make_action_plan_dto()


@pytest.fixture
def fake_usage_log() -> UsageLogDTO:
    return make_usage_log_dto()


@pytest.fixture
def valid_token() -> str:
    return create_access_token("user-uuid-1")


@pytest.fixture
def mock_db() -> AsyncMock:
    session = AsyncMock()
    # Explicitly use MagicMock as return_value so that result.scalar_one_or_none()
    # etc. return sync values (not coroutines) — needed in Python 3.13.
    session.execute = AsyncMock(return_value=MagicMock())
    session.add = MagicMock()
    session.commit = AsyncMock()
    session.refresh = AsyncMock()
    session.delete = AsyncMock()
    session.flush = AsyncMock()
    return session
