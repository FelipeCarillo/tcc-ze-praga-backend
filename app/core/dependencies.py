from collections.abc import AsyncGenerator

from fastapi import Depends
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import ForbiddenError, UnauthorizedError
from app.core.security import decode_access_token
from app.db.database import AsyncSessionLocal
from app.domains.auth.api_key_repository import ApiKeyRepository
from app.domains.auth.api_key_service import ApiKeyService
from app.domains.auth.dto import UserDTO
from app.domains.auth.repository import UserRepository
from app.domains.auth.service import AuthService
from app.shared.enums import FeatureTypeEnum

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


# ── Database ──────────────────────────────────────────────────────────────────


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session


# ── Repositories ──────────────────────────────────────────────────────────────


def get_user_repository(db: AsyncSession = Depends(get_db)) -> UserRepository:
    return UserRepository(db)


def get_subscription_repository(db: AsyncSession = Depends(get_db)):  # type: ignore[no-untyped-def]
    from app.domains.subscriptions.repository import SubscriptionRepository

    return SubscriptionRepository(db)


def get_usage_repository(db: AsyncSession = Depends(get_db)):  # type: ignore[no-untyped-def]
    from app.domains.usage.repository import UsageRepository

    return UsageRepository(db)


def get_diagnosis_repository(db: AsyncSession = Depends(get_db)):  # type: ignore[no-untyped-def]
    from app.domains.diagnoses.repository import DiagnosisRepository

    return DiagnosisRepository(db)


def get_action_plan_repository(db: AsyncSession = Depends(get_db)):  # type: ignore[no-untyped-def]
    from app.domains.action_plans.repository import ActionPlanRepository

    return ActionPlanRepository(db)


def get_chat_session_repository(db: AsyncSession = Depends(get_db)):  # type: ignore[no-untyped-def]
    from app.domains.chat.repository import ChatSessionRepository

    return ChatSessionRepository(db)


def get_chat_message_repository(db: AsyncSession = Depends(get_db)):  # type: ignore[no-untyped-def]
    from app.domains.chat.repository import ChatMessageRepository

    return ChatMessageRepository(db)


def get_crop_repository(db: AsyncSession = Depends(get_db)):  # type: ignore[no-untyped-def]
    from app.domains.inference.repository import CropRepository

    return CropRepository(db)


def get_disease_repository(db: AsyncSession = Depends(get_db)):  # type: ignore[no-untyped-def]
    from app.domains.inference.repository import DiseaseRepository

    return DiseaseRepository(db)


def get_uploaded_file_repository(db: AsyncSession = Depends(get_db)):  # type: ignore[no-untyped-def]
    from app.domains.uploads.repository import UploadedFileRepository

    return UploadedFileRepository(db)


# ── Store (long-term memory) ──────────────────────────────────────────────────


async def get_store_dep():  # type: ignore[no-untyped-def]
    """Dependency-injection wrapper para o singleton ``AsyncPostgresStore``."""
    from app.db.store import get_store

    return await get_store()


# ── Services ──────────────────────────────────────────────────────────────────


def get_upload_service(  # type: ignore[no-untyped-def]
    repo=Depends(get_uploaded_file_repository),
):
    """UploadService com SupabaseStorageUploader (lru_cache no client)."""
    from app.db.storage import get_storage_client
    from app.domains.uploads.service import SupabaseStorageUploader, UploadService

    uploader = SupabaseStorageUploader(get_storage_client())
    return UploadService(repo, uploader)


def get_auth_service(
    repo: UserRepository = Depends(get_user_repository),
) -> AuthService:
    return AuthService(repo)


def get_api_key_repository(db: AsyncSession = Depends(get_db)) -> ApiKeyRepository:
    return ApiKeyRepository(db)


def get_api_key_service(
    repo: ApiKeyRepository = Depends(get_api_key_repository),
) -> ApiKeyService:
    return ApiKeyService(repo)


def get_user_service(  # type: ignore[no-untyped-def]
    repo: UserRepository = Depends(get_user_repository),
    sub_repo=Depends(get_subscription_repository),
):
    from app.domains.users.service import UserService

    return UserService(repo, sub_repo)


def get_subscription_service(repo=Depends(get_subscription_repository)):  # type: ignore[no-untyped-def]
    from app.domains.subscriptions.service import SubscriptionService

    return SubscriptionService(repo)


def get_usage_service(  # type: ignore[no-untyped-def]
    usage_repo=Depends(get_usage_repository),
    sub_repo=Depends(get_subscription_repository),
):
    from app.domains.usage.service import UsageService

    return UsageService(usage_repo, sub_repo)


def get_diagnosis_service(  # type: ignore[no-untyped-def]
    repo=Depends(get_diagnosis_repository),
):
    from app.domains.diagnoses.service import DiagnosisService

    return DiagnosisService(repo)


def get_action_plan_service(repo=Depends(get_action_plan_repository)):  # type: ignore[no-untyped-def]
    from app.domains.action_plans.service import ActionPlanService

    return ActionPlanService(repo)


async def get_inference_service(  # type: ignore[no-untyped-def]
    crop_repo=Depends(get_crop_repository),
    disease_repo=Depends(get_disease_repository),
):
    """Carrega o catalogo de doencas da soja e instancia o InferenceService.

    Usa cache do DiseaseRepository — chamada repetida nao bate no DB.
    O catalogo e' carregado pra crop ``soja`` por padrao (multi-cultivo
    completo vem em sprint A2).
    """
    from app.domains.inference.service import InferenceService

    crop = await crop_repo.get_by_slug("soja")
    if crop is None:
        raise RuntimeError(
            "Crop 'soja' nao encontrada no DB — rode `uv run python -m scripts.seed_crops`."
        )
    diseases = await disease_repo.list_by_crop(crop.id)
    return InferenceService(diseases=diseases)


async def get_inference_service_for_crop(  # type: ignore[no-untyped-def]
    crop_id_or_slug: str,
    crop_repo,
    disease_repo,
):
    """Carrega InferenceService para um crop especifico (slug ou id).

    Usado pelo factory do sub-grafo (TCC-042) — multi-cultivo runtime.
    """
    from app.domains.inference.service import InferenceService

    crop = await crop_repo.get_by_slug(crop_id_or_slug)
    if crop is None:
        crop = await crop_repo.get_by_id(crop_id_or_slug)
    if crop is None:
        raise RuntimeError(
            f"Crop '{crop_id_or_slug}' nao encontrada — verifique seed_crops."
        )
    diseases = await disease_repo.list_by_crop(crop.id)
    return InferenceService(diseases=diseases)


def get_diagnosis_graph_factory(  # type: ignore[no-untyped-def]
    inference_svc=Depends(get_inference_service),
    action_plan_svc=Depends(get_action_plan_service),
    diagnosis_svc=Depends(get_diagnosis_service),
):
    """Factory cacheada (por request) de ``CompiledStateGraph`` do diagnosis_graph.

    Em Sprint A2 o ``inference_svc`` ja vem com catalogo da soja (default
    do ``get_inference_service``). Multi-cultivo verdadeiro entra quando
    o factory carregar diseases dinamicamente por ``crop_id`` — por ora o
    closure ignora ``crop_id`` recebido e usa o svc injetado.

    Em Sprint A2.5 o factory aceita ``store`` opcional pra indexar
    diagnoses no Store; quando ``None``, fallback pra build sem store
    (back-compat).
    """
    from app.domains.diagnosis_graph.graph import build_diagnosis_graph

    _cache: dict[str, object] = {}

    def _factory(crop_id: str, store=None):  # type: ignore[no-untyped-def]
        cache_key = f"{crop_id}::{id(store) if store else 'no-store'}"
        if cache_key in _cache:
            return _cache[cache_key]
        graph = build_diagnosis_graph(
            inference_svc, action_plan_svc, diagnosis_svc, store=store
        )
        _cache[cache_key] = graph
        return graph

    return _factory


def get_chat_service(  # type: ignore[no-untyped-def]
    session_repo=Depends(get_chat_session_repository),
    message_repo=Depends(get_chat_message_repository),
    inference_svc=Depends(get_inference_service),
    action_plan_svc=Depends(get_action_plan_service),
    diagnosis_svc=Depends(get_diagnosis_service),
    sub_repo=Depends(get_subscription_repository),
):
    from app.db.store import get_store
    from app.domains.chat.service import ChatService

    return ChatService(
        session_repo=session_repo,
        message_repo=message_repo,
        inference_svc=inference_svc,
        action_plan_svc=action_plan_svc,
        diagnosis_svc=diagnosis_svc,
        store_factory=get_store,
        sub_repo=sub_repo,
    )


# ── Auth ──────────────────────────────────────────────────────────────────────


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    repo: UserRepository = Depends(get_user_repository),
) -> UserDTO:
    payload = decode_access_token(token)
    user = await repo.find_by_id(payload["sub"])
    if not user or not user.is_active:
        raise UnauthorizedError("User not found or inactive")
    return user


# ── Quota ─────────────────────────────────────────────────────────────────────


def require_quota(feature: FeatureTypeEnum):  # type: ignore[no-untyped-def]
    async def _dependency(
        current_user: UserDTO = Depends(get_current_user),
        usage_svc=Depends(get_usage_service),
    ) -> UserDTO:
        await usage_svc.check_quota(current_user.id, feature)
        return current_user

    return _dependency


# ── Tier ──────────────────────────────────────────────────────────────────────


async def require_tier_enterprise(  # type: ignore[no-untyped-def]
    current_user: UserDTO = Depends(get_current_user),
    sub_repo=Depends(get_subscription_repository),
) -> UserDTO:
    """Permite acesso apenas a usuarios com plano Enterprise (api_access=True).

    Checa pela feature flag ``api_access`` no JSON ``features`` do plano —
    fallback pra tier_name == 'enterprise' quando features for None.
    """
    sub = await sub_repo.find_user_subscription(current_user.id)
    if not sub:
        raise ForbiddenError("API keys disponiveis apenas no plano Enterprise")

    features = sub.plan.features or {}
    api_access = features.get("api_access") if isinstance(features, dict) else False
    if api_access:
        return current_user

    # Fallback pre-features (em testes/seeds antigos): aceita pelo nome.
    if sub.plan.name == "enterprise":
        return current_user

    raise ForbiddenError("API keys disponiveis apenas no plano Enterprise")
