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
    from app.domains.uploads.service import UploadService

    repo = MagicMock(spec=DiagnosisRepository)
    upload_svc = MagicMock(spec=UploadService)
    result = get_diagnosis_service(repo, upload_svc)
    assert isinstance(result, DiagnosisService)


def test_get_diagnosis_service_recebe_resolvedor_de_imagem():
    """Sem o resolvedor, ``image_url`` sairia como storage key crua na resposta."""
    from app.core.dependencies import get_diagnosis_service
    from app.domains.diagnoses.repository import DiagnosisRepository
    from app.domains.uploads.service import UploadService

    repo = MagicMock(spec=DiagnosisRepository)
    upload_svc = MagicMock(spec=UploadService)
    svc = get_diagnosis_service(repo, upload_svc)
    assert svc._resolve_image_urls is upload_svc.signed_urls


def test_get_action_plan_service():
    from app.core.dependencies import get_action_plan_service
    from app.domains.action_plans.repository import ActionPlanRepository
    from app.domains.action_plans.service import ActionPlanService

    repo = MagicMock(spec=ActionPlanRepository)
    result = get_action_plan_service(repo)
    assert isinstance(result, ActionPlanService)


async def test_get_inference_service():
    """Agora async — recebe CropRepository/DiseaseRepository por DI."""
    from app.core.dependencies import get_inference_service
    from app.domains.inference.repository import (
        CropDTO,
        CropRepository,
        DiseaseDTO,
        DiseaseRepository,
    )
    from app.domains.inference.service import InferenceService

    crop_dto = CropDTO(
        id="soja-id",
        slug="soja",
        name_pt="Soja",
        scientific_name="Glycine max",
        kingdom="Plantae",
        is_active=True,
    )
    disease_dto = DiseaseDTO(
        id="d-1",
        crop_id="soja-id",
        slug="ferrugem-asiatica",
        name_pt="Ferrugem Asiática",
        scientific_name="Phakopsora pachyrhizi",
        severity_default="alta",
        description_md="Desc",
        image_url=None,
    )

    crop_repo = MagicMock(spec=CropRepository)
    crop_repo.get_by_slug = AsyncMock(return_value=crop_dto)
    disease_repo = MagicMock(spec=DiseaseRepository)
    disease_repo.list_by_crop = AsyncMock(return_value=[disease_dto])

    result = await get_inference_service(crop_repo=crop_repo, disease_repo=disease_repo)
    assert isinstance(result, InferenceService)
    crop_repo.get_by_slug.assert_awaited_once_with("soja")
    disease_repo.list_by_crop.assert_awaited_once_with("soja-id")


async def test_get_inference_service_raises_when_soja_missing():
    """Sem crop 'soja' no DB, factory levanta RuntimeError."""
    from app.core.dependencies import get_inference_service
    from app.domains.inference.repository import CropRepository, DiseaseRepository

    crop_repo = MagicMock(spec=CropRepository)
    crop_repo.get_by_slug = AsyncMock(return_value=None)
    disease_repo = MagicMock(spec=DiseaseRepository)

    with pytest.raises(RuntimeError, match="seed_crops"):
        await get_inference_service(crop_repo=crop_repo, disease_repo=disease_repo)


def test_get_crop_repository():
    from app.core.dependencies import get_crop_repository
    from app.domains.inference.repository import CropRepository

    db = MagicMock(spec=AsyncSession)
    result = get_crop_repository(db)
    assert isinstance(result, CropRepository)


def test_get_disease_repository():
    from app.core.dependencies import get_disease_repository
    from app.domains.inference.repository import DiseaseRepository

    db = MagicMock(spec=AsyncSession)
    result = get_disease_repository(db)
    assert isinstance(result, DiseaseRepository)


def test_get_chat_session_repository():
    from app.core.dependencies import get_chat_session_repository
    from app.domains.chat.repository import ChatSessionRepository

    db = MagicMock(spec=AsyncSession)
    result = get_chat_session_repository(db)
    assert isinstance(result, ChatSessionRepository)


def test_get_chat_message_repository():
    from app.core.dependencies import get_chat_message_repository
    from app.domains.chat.repository import ChatMessageRepository

    db = MagicMock(spec=AsyncSession)
    result = get_chat_message_repository(db)
    assert isinstance(result, ChatMessageRepository)


def test_get_chat_service():
    from app.core.dependencies import get_chat_service
    from app.domains.action_plans.service import ActionPlanService
    from app.domains.chat.repository import ChatMessageRepository, ChatSessionRepository
    from app.domains.chat.service import ChatService
    from app.domains.diagnoses.service import DiagnosisService
    from app.domains.inference.service import InferenceService

    session_repo = MagicMock(spec=ChatSessionRepository)
    message_repo = MagicMock(spec=ChatMessageRepository)
    inference_svc = MagicMock(spec=InferenceService)
    action_plan_svc = MagicMock(spec=ActionPlanService)
    diagnosis_svc = MagicMock(spec=DiagnosisService)

    result = get_chat_service(
        session_repo=session_repo,
        message_repo=message_repo,
        inference_svc=inference_svc,
        action_plan_svc=action_plan_svc,
        diagnosis_svc=diagnosis_svc,
    )
    assert isinstance(result, ChatService)


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
