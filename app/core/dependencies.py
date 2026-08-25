from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import TYPE_CHECKING, Any

from fastapi import Depends, Header
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import ForbiddenError, UnauthorizedError
from app.core.security import decode_access_token
from app.db.database import AsyncSessionLocal
from app.domains.auth.api_key_repository import ApiKeyRepository
from app.domains.auth.api_key_service import ApiKeyService
from app.domains.auth.dto import UserDTO
from app.domains.auth.repository import (
    EmailVerificationRepository,
    PasswordResetRepository,
    UserRepository,
)
from app.domains.auth.service import AuthService
from app.shared.enums import FeatureTypeEnum

if TYPE_CHECKING:
    from app.domains.inference.onnx_classifier import OnnxClassifier
    from app.domains.usage.service import UsageService

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


# ── Checkpointer (HITL — TCC-058) ─────────────────────────────────────────────


async def get_checkpointer_dep():  # type: ignore[no-untyped-def]
    """Dependency-injection wrapper para o singleton ``AsyncPostgresSaver``."""
    from app.db.checkpointer import get_checkpointer

    return await get_checkpointer()


# ── Services ──────────────────────────────────────────────────────────────────


def get_upload_service(  # type: ignore[no-untyped-def]
    repo=Depends(get_uploaded_file_repository),
):
    """UploadService com SupabaseStorageUploader (lru_cache no client)."""
    from app.db.storage import get_storage_client
    from app.domains.uploads.service import SupabaseStorageUploader, UploadService

    uploader = SupabaseStorageUploader(get_storage_client())
    return UploadService(repo, uploader)


def get_email_verification_repository(
    db: AsyncSession = Depends(get_db),
) -> EmailVerificationRepository:
    return EmailVerificationRepository(db)


def get_password_reset_repository(
    db: AsyncSession = Depends(get_db),
) -> PasswordResetRepository:
    return PasswordResetRepository(db)


def get_auth_service(
    repo: UserRepository = Depends(get_user_repository),
    verification_repo: EmailVerificationRepository = Depends(get_email_verification_repository),
    reset_repo: PasswordResetRepository = Depends(get_password_reset_repository),
) -> AuthService:
    # O sender fica None: o service resolve via get_email_sender() na hora do
    # envio, caindo no NullEmailSender quando nao ha RESEND_API_KEY (TCC-090).
    return AuthService(repo, verification_repo, reset_repo=reset_repo)


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
    upload_svc=Depends(get_upload_service),
):
    """DiagnosisService com resolvedor de imagem.

    ``diagnoses.image_url`` guarda a storage key; o resolvedor a converte em URL
    assinada na leitura (em lote). Sem ele o campo sai cru e a UI nao renderiza.
    """
    from app.domains.diagnoses.service import DiagnosisService

    return DiagnosisService(repo, image_url_resolver=upload_svc.signed_urls)


def get_action_plan_service(repo=Depends(get_action_plan_repository)):  # type: ignore[no-untyped-def]
    from app.domains.action_plans.service import ActionPlanService

    return ActionPlanService(repo)


def get_talhao_repository(db: AsyncSession = Depends(get_db)):  # type: ignore[no-untyped-def]
    from app.domains.talhoes.repository import TalhaoRepository

    return TalhaoRepository(db)


def get_talhao_service(repo=Depends(get_talhao_repository)):  # type: ignore[no-untyped-def]
    from app.domains.talhoes.service import TalhaoService

    return TalhaoService(repo)


# Registro multi-modelo (TCC-095): cada chave canônica → (path relativo, input_size).
# A ordem das classes é a mesma (alfabética, label_map.csv) p/ os 3 modelos, então
# cada .onnx usa o .labels.json irmão idêntico. Permite troca real de modelo +
# ensemble no chat e no REST /inference.
_ONNX_MODELS: dict[str, tuple[str, int]] = {
    "efficientnet_b4": ("models/soja_efficientnet_b4.onnx", 380),
    "resnet50": ("models/soja_resnet50.onnx", 224),
    "vit_b16": ("models/soja_vit_b16.onnx", 224),
}

# Cache do registro de OnnxClassifiers — carregado uma vez por processo.
# _onnx_loaded evita recarregar os .onnx a cada request. TCC-023/TCC-095, ADR-0003.
_onnx_classifiers: "dict[str, OnnxClassifier]" = {}
_onnx_loaded = False


def _get_onnx_classifiers() -> "dict[str, OnnxClassifier]":
    """Retorna ``{chave_canônica: OnnxClassifier}`` para os modelos disponíveis.

    Graceful: flag off → vazio; cada modelo ausente ou que falhe ao carregar é
    apenas pulado (o InferenceService cai no mock/default p/ aquele id).
    """
    global _onnx_classifiers, _onnx_loaded
    if _onnx_loaded:
        return _onnx_classifiers
    _onnx_loaded = True

    import logging
    from pathlib import Path

    from app.config import settings

    log = logging.getLogger(__name__)
    if not settings.inference_use_onnx:
        return _onnx_classifiers

    repo_root = Path(__file__).resolve().parents[2]
    for key, (rel_path, input_size) in _ONNX_MODELS.items():
        try:
            from app.domains.inference.onnx_classifier import OnnxClassifier

            path = Path(rel_path)
            if not path.is_absolute():
                path = repo_root / path
            if not path.exists():
                log.warning("Modelo ONNX '%s' não encontrado em %s — pulado", key, path)
                continue
            _onnx_classifiers[key] = OnnxClassifier.from_path(path, input_size=input_size)
            log.info("OnnxClassifier carregado: %s (%s, %dpx)", key, path.name, input_size)
        except Exception:
            log.exception("Falha ao carregar OnnxClassifier '%s' — pulado", key)
    return _onnx_classifiers


def _get_onnx_classifier() -> "OnnxClassifier | None":
    """Default (efficientnet_b4) — back-compat para callers de modelo único."""
    return _get_onnx_classifiers().get("efficientnet_b4")


async def get_inference_service(  # type: ignore[no-untyped-def]
    crop_repo=Depends(get_crop_repository),
    disease_repo=Depends(get_disease_repository),
):
    """Carrega o catalogo de doencas da soja e instancia o InferenceService.

    Usa cache do DiseaseRepository — chamada repetida nao bate no DB.
    O catalogo e' carregado pra crop ``soja`` por padrao (multi-cultivo
    completo vem em sprint A2). Injeta o OnnxClassifier real (TCC-023) quando
    disponível; senão o service usa o mock.
    """
    from app.domains.inference.service import InferenceService

    crop = await crop_repo.get_by_slug("soja")
    if crop is None:
        raise RuntimeError(
            "Crop 'soja' nao encontrada no DB — rode `uv run python -m scripts.seed_crops`."
        )
    diseases = await disease_repo.list_by_crop(crop.id)
    return InferenceService(diseases=diseases, classifiers=_get_onnx_classifiers())


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
    return InferenceService(diseases=diseases, classifiers=_get_onnx_classifiers())


def get_diagnosis_graph_factory(  # type: ignore[no-untyped-def]
    inference_svc=Depends(get_inference_service),
    action_plan_svc=Depends(get_action_plan_service),
    diagnosis_svc=Depends(get_diagnosis_service),
    upload_svc=Depends(get_upload_service),
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
            inference_svc,
            action_plan_svc,
            diagnosis_svc,
            store=store,
            upload_svc=upload_svc,
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
    diagnosis_graph_factory=Depends(get_diagnosis_graph_factory),
    upload_svc=Depends(get_upload_service),
):
    """ChatService com todas as deps do registry de tools.

    ``diagnosis_graph_factory`` habilita ``deep_diagnose`` e ``compare_diagnoses``
    — sem ela essas duas tools ficam de fora do grafo (ver
    ``ChatService._build_tool_factories``).
    """
    from app.db.checkpointer import get_checkpointer
    from app.db.store import get_store
    from app.domains.chat.service import ChatService

    return ChatService(
        session_repo=session_repo,
        message_repo=message_repo,
        inference_svc=inference_svc,
        action_plan_svc=action_plan_svc,
        diagnosis_svc=diagnosis_svc,
        store_factory=get_store,
        checkpointer_factory=get_checkpointer,
        sub_repo=sub_repo,
        diagnosis_graph_factory=diagnosis_graph_factory,
        upload_svc=upload_svc,
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


async def get_current_user_or_api_key(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    user_repo: UserRepository = Depends(get_user_repository),
    api_key_svc: ApiKeyService = Depends(get_api_key_service),
) -> UserDTO:
    """Auth dual — resolve user via JWT ``Authorization: Bearer`` OU via
    header ``X-API-Key: zp_live_...``.

    Precedence: ``X-API-Key`` ganha. Pra endpoints publicos como
    ``POST /diagnoses/analyze`` que sao expostos a Enterprise via API REST.

    Fluxo X-API-Key:
        1. Verify (lookup por prefix + bcrypt checkpw)
        2. Carrega user dono da key
        3. ``touch_last_used`` async
        4. Retorna ``UserDTO`` ativo

    Fluxo JWT: identico ao ``get_current_user``.
    """
    if x_api_key:
        api_key = await api_key_svc.verify(x_api_key)
        if api_key is None or not api_key.is_active:
            raise UnauthorizedError("Invalid API key")
        await api_key_svc.touch_last_used(api_key.id)
        user = await user_repo.find_by_id(api_key.user_id)
        if user is None or not user.is_active:
            raise UnauthorizedError("API key user not found or inactive")
        return user

    if authorization and authorization.startswith("Bearer "):
        token = authorization.replace("Bearer ", "", 1)
        payload = decode_access_token(token)
        user = await user_repo.find_by_id(payload["sub"])
        if not user or not user.is_active:
            raise UnauthorizedError("User not found or inactive")
        return user

    raise UnauthorizedError("Authentication required")


async def auth_method_dual(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> str:
    """Auxiliar leve pra descobrir como o request foi autenticado.

    Retorna ``"api_key"`` quando ``X-API-Key`` esta presente,
    senao ``"jwt"``. Usado por handlers que precisam contabilizar quota
    diferente (INFERENCE vs API) e por middleware/rate-limit headers.
    """
    return "api_key" if x_api_key else "jwt"


# ── Plan features ─────────────────────────────────────────────────────────────


def plan_features_dep(  # type: ignore[no-untyped-def]
    user_dep: Callable[..., Awaitable[UserDTO]],
):
    """Factory de dependency que resolve ``PlanFeatures`` do plano ativo.

    Recebe a dependency de auth que a rota ja usa (``get_current_user`` pras
    rotas so'-JWT, ``get_current_user_or_api_key`` pras de auth dual) em vez de
    resolver o usuario por conta propria. Assim o FastAPI reaproveita o cache
    de dependencia do request — sem segundo decode de token — e os overrides
    de auth nos testes continuam valendo.

    Centraliza o que o ``ChatService._resolve_plan_features`` faz, pros routers
    REST aplicarem os mesmos gates (ex: ``diagnosis_models``) sem duplicar a
    resolucao da subscription.
    """

    async def _dependency(  # type: ignore[no-untyped-def]
        current_user: UserDTO = Depends(user_dep),
        sub_repo=Depends(get_subscription_repository),
    ):
        from app.domains.subscriptions.features import FREE_FEATURES, PlanFeatures

        sub = await sub_repo.find_user_subscription(current_user.id)
        if sub is None or sub.plan.features is None:
            return FREE_FEATURES
        try:
            return PlanFeatures(**sub.plan.features)
        except Exception:  # noqa: BLE001 — features malformadas nao travam o request
            return FREE_FEATURES

    return _dependency


# Variantes prontas — uma por caminho de auth.
get_plan_features = plan_features_dep(get_current_user)
get_plan_features_dual = plan_features_dep(get_current_user_or_api_key)


# ── Quota ─────────────────────────────────────────────────────────────────────


def require_quota(feature: FeatureTypeEnum) -> Callable[..., Awaitable[UserDTO]]:
    async def _dependency(
        current_user: UserDTO = Depends(get_current_user),
        usage_svc: "UsageService" = Depends(get_usage_service),
    ) -> UserDTO:
        await usage_svc.check_quota(current_user.id, feature)
        return current_user

    return _dependency


async def require_quota_dual(  # type: ignore[no-untyped-def]
    current_user: UserDTO = Depends(get_current_user_or_api_key),
    auth_method: str = Depends(auth_method_dual),
    usage_svc=Depends(get_usage_service),
) -> UserDTO:
    """Quota-check para endpoints com auth dual (JWT ou API key).

    Mapeia auth method -> feature:
        - ``api_key`` -> ``FeatureTypeEnum.API`` (monthly quota)
        - ``jwt``     -> ``FeatureTypeEnum.INFERENCE`` (daily quota)
    """
    feature = (
        FeatureTypeEnum.API if auth_method == "api_key" else FeatureTypeEnum.INFERENCE
    )
    await usage_svc.check_quota(current_user.id, feature)
    return current_user


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


# ── Transcription / STT (TCC-081) ───────────────────────────────────────────────

_transcription_service: Any = None


def get_transcription_service():  # type: ignore[no-untyped-def]
    """Singleton do ``TranscriptionService`` (OpenAI STT) — entrada de voz no chat."""
    from app.domains.transcription.service import TranscriptionService

    global _transcription_service
    if _transcription_service is None:
        _transcription_service = TranscriptionService()
    return _transcription_service
