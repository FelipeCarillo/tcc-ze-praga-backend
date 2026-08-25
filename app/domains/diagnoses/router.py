import base64
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Response,
    UploadFile,
    status,
)

from app.core.dependencies import (
    auth_method_dual,
    get_current_user,
    get_diagnosis_graph_factory,
    get_diagnosis_repository,
    get_diagnosis_service,
    get_inference_service,
    get_plan_features_dual,
    get_store_dep,
    get_subscription_repository,
    get_usage_repository,
    get_usage_service,
    require_quota,
    require_quota_dual,
)
from app.core.exceptions import NotFoundError
from app.domains.auth.dto import UserDTO
from app.domains.chat.schemas import SemanticDiagnosisHit
from app.domains.diagnoses.schemas import (
    CreateDiagnosisRequest,
    DiagnosisFilters,
    DiagnosisResponse,
    Top3PredictionSchema,
)
from app.domains.diagnoses.service import DiagnosisService
from app.domains.inference.service import InferenceService, resolve_allowed_model
from app.domains.subscriptions.features import PlanFeatures
from app.domains.usage.service import UsageService
from app.shared.enums import FeatureTypeEnum, SeverityEnum
from app.shared.pagination import PaginatedResponse

if TYPE_CHECKING:
    from langgraph.store.postgres.aio import AsyncPostgresStore

    from app.domains.diagnoses.repository import DiagnosisRepository
    from app.domains.subscriptions.repository import SubscriptionRepository
    from app.domains.usage.repository import UsageRepository

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/diagnoses", tags=["Diagnoses"])

# Espelha o limite do chat (app/domains/chat/router.py) — mesma politica de upload.
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB


async def _read_image_batch(
    images: list[UploadFile],
) -> tuple[list[str], list[str]]:
    """Le os uploads e devolve ``(image_batch_b64, image_ids)`` alinhados.

    Sem isto o sub-grafo recebia ``image_batch=[]`` e o
    ``run_inference_node`` caia no mock — os bytes eram lidos e descartados.
    """
    batch: list[str] = []
    ids: list[str] = []
    for i, img in enumerate(images):
        data = await img.read()
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=(
                    f"Imagem '{img.filename or i}' excede o limite de "
                    f"{MAX_UPLOAD_BYTES // (1024 * 1024)}MB."
                ),
            )
        batch.append(base64.b64encode(data).decode("ascii"))
        ids.append(img.filename or f"image-{i}")
    return batch, ids


@router.post("/analyze", response_model=list[DiagnosisResponse], status_code=200)
async def analyze(
    response: Response,
    images: list[UploadFile] = File(...),
    crop_id: str = Form(default="soja"),
    model: str = Form(default="ensemble"),
    current_user: UserDTO = Depends(require_quota_dual),
    auth_method: str = Depends(auth_method_dual),
    diagnosis_graph_factory: Callable[[str], Any] = Depends(get_diagnosis_graph_factory),
    diag_repo: "DiagnosisRepository" = Depends(get_diagnosis_repository),
    usage_svc: UsageService = Depends(get_usage_service),
    usage_repo: "UsageRepository" = Depends(get_usage_repository),
    sub_repo: "SubscriptionRepository" = Depends(get_subscription_repository),
    plan_features: PlanFeatures = Depends(get_plan_features_dual),
) -> list[DiagnosisResponse]:
    """Endpoint REST direto pra diagnostico (sem chat).

    Auth dual: aceita JWT (web) OU ``X-API-Key`` (Enterprise via REST). Quota
    eh contada por INFERENCE (daily) com JWT, ou API (monthly) com API key.
    Quando autenticado via API key, response carrega headers
    ``X-RateLimit-{Limit,Remaining,Reset}`` (TCC-064).
    """
    feature = (
        FeatureTypeEnum.API if auth_method == "api_key" else FeatureTypeEnum.INFERENCE
    )

    graph = diagnosis_graph_factory(crop_id)
    image_batch, image_ids = await _read_image_batch(images)
    effective_model, _downgraded = resolve_allowed_model(
        model, plan_features.diagnosis_models
    )

    result = await graph.ainvoke(
        {
            "user_id": current_user.id,
            "crop_id": crop_id,
            "image_batch": image_batch,
            "image_ids": image_ids,
            "model_id": effective_model,
            "plan_features": plan_features.model_dump(),
        }
    )

    persisted_ids = result.get("persisted_ids", [])
    responses: list[DiagnosisResponse] = []
    for diag_id in persisted_ids:
        dto = await diag_repo.find_by_id(diag_id, current_user.id)
        if not dto:
            raise NotFoundError("Diagnosis", diag_id)
        responses.append(
            DiagnosisResponse(
                id=dto.id,
                disease_name=dto.disease_name,
                disease_id=dto.disease_id,
                scientific_name=dto.scientific_name,
                confidence=dto.confidence,
                severity=dto.severity,
                description=dto.description,
                model_used=dto.model_used,
                image_url=dto.image_url,
                image_name=dto.image_name,
                created_at=dto.created_at,
                top3=[
                    Top3PredictionSchema(
                        rank=t.rank,
                        disease_name=t.disease_name,
                        disease_id=t.disease_id,
                        scientific_name=t.scientific_name,
                        confidence=t.confidence,
                        severity=t.severity,
                    )
                    for t in dto.top3
                ],
            )
        )

    await usage_svc.record_usage(
        current_user.id,
        feature,
        {
            "crop_id": crop_id,
            "model": effective_model,
            "model_requested": model,
            "batch_size": len(images),
            "auth_method": auth_method,
        },
    )

    # TCC-064: rate limit headers — so' faz sentido pra API key (monthly quota).
    if auth_method == "api_key":
        await _set_api_rate_limit_headers(
            response, current_user.id, usage_repo, sub_repo
        )

    return responses


async def _set_api_rate_limit_headers(
    response: Response,
    user_id: str,
    usage_repo: "UsageRepository",
    sub_repo: "SubscriptionRepository",
) -> None:
    """Anexa X-RateLimit-* na response do endpoint /analyze quando autenticado via API.

    Headers (espelham padrao IETF draft):
        X-RateLimit-Limit:     limite mensal do plano (api_monthly_limit)
        X-RateLimit-Remaining: limit - usado no mes corrente
        X-RateLimit-Reset:     epoch UTC do inicio do proximo mes
    """
    import calendar
    from datetime import UTC, datetime, timedelta

    sub = await sub_repo.find_user_subscription(user_id)
    limit = (
        sub.plan.api_monthly_limit if sub and sub.plan.api_monthly_limit is not None else 0
    )
    used = await usage_repo.count_this_month(user_id, FeatureTypeEnum.API)
    remaining = max(0, limit - used) if limit else 0

    now = datetime.now(UTC)
    # Ultimo dia do mes corrente
    _, last_day = calendar.monthrange(now.year, now.month)
    next_month_start = (
        datetime(now.year, now.month, last_day, 23, 59, 59, tzinfo=UTC) + timedelta(seconds=1)
    )
    reset_epoch = int(next_month_start.timestamp())

    response.headers["X-RateLimit-Limit"] = str(limit)
    response.headers["X-RateLimit-Remaining"] = str(remaining)
    response.headers["X-RateLimit-Reset"] = str(reset_epoch)


@router.post("", response_model=DiagnosisResponse, status_code=201)
async def create_diagnosis(
    body: CreateDiagnosisRequest,
    current_user: UserDTO = Depends(require_quota(FeatureTypeEnum.INFERENCE)),
    service: DiagnosisService = Depends(get_diagnosis_service),
    usage_svc: UsageService = Depends(get_usage_service),
    inference_svc: InferenceService = Depends(get_inference_service),
) -> DiagnosisResponse:
    crop_uuid = (
        inference_svc.disease_catalog[0].crop_id
        if inference_svc.disease_catalog
        else ""
    )
    result = await service.create(current_user.id, body, crop_id=crop_uuid)
    await usage_svc.record_usage(
        current_user.id,
        FeatureTypeEnum.INFERENCE,
        {"disease_id": body.disease_id, "model": body.model_used},
    )
    return result


@router.get("", response_model=PaginatedResponse[DiagnosisResponse])
async def list_diagnoses(
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=20, ge=1, le=100),
    severity: SeverityEnum | None = Query(default=None),
    search: str | None = Query(default=None),
    current_user: UserDTO = Depends(get_current_user),
    service: DiagnosisService = Depends(get_diagnosis_service),
) -> PaginatedResponse[DiagnosisResponse]:
    filters = DiagnosisFilters(page=page, limit=limit, severity=severity, search=search)
    return await service.list_for_user(current_user.id, filters)


@router.get("/semantic", response_model=list[SemanticDiagnosisHit])
async def search_diagnoses_semantic(
    q: str = Query(..., min_length=1, description="Texto da busca semantica"),
    limit: int = Query(default=5, ge=1, le=50),
    current_user: UserDTO = Depends(get_current_user),
    store: "AsyncPostgresStore" = Depends(get_store_dep),
) -> list[SemanticDiagnosisHit]:
    """Busca semantica em diagnoses passados via Store (TCC-048).

    Consulta o namespace ``("user", uid, "diagnoses")`` por similaridade
    de embedding com o ``q``.
    """
    try:
        results = await store.asearch(
            ("user", current_user.id, "diagnoses"),
            query=q,
            limit=limit,
        )
    except Exception:  # noqa: BLE001
        logger.exception("Semantic search failed for user %s", current_user.id)
        return []

    hits: list[SemanticDiagnosisHit] = []
    for r in results:
        value = r.value if isinstance(r.value, dict) else dict(r.value)
        hits.append(
            SemanticDiagnosisHit(
                summary_text=value.get("summary_text", ""),
                diagnosis_id=value.get("diagnosis_id", ""),
                disease_id=value.get("disease_id"),
                disease_name=value.get("disease_name"),
                crop_id=value.get("crop_id"),
                confidence=value.get("confidence"),
                severity=value.get("severity"),
                created_at=value.get("created_at"),
            )
        )
    return hits


@router.get("/{diagnosis_id}", response_model=DiagnosisResponse)
async def get_diagnosis(
    diagnosis_id: str,
    current_user: UserDTO = Depends(get_current_user),
    service: DiagnosisService = Depends(get_diagnosis_service),
) -> DiagnosisResponse:
    return await service.get_by_id(diagnosis_id, current_user.id)


@router.delete("/{diagnosis_id}", status_code=204)
async def delete_diagnosis(
    diagnosis_id: str,
    current_user: UserDTO = Depends(get_current_user),
    service: DiagnosisService = Depends(get_diagnosis_service),
) -> None:
    await service.delete(diagnosis_id, current_user.id)


@router.delete("", status_code=200)
async def clear_all_diagnoses(
    confirm: bool = Query(default=False),
    current_user: UserDTO = Depends(get_current_user),
    service: DiagnosisService = Depends(get_diagnosis_service),
) -> dict[str, Any]:
    if not confirm:
        return {"detail": "Pass ?confirm=true to delete all diagnoses"}
    deleted = await service.clear_all(current_user.id)
    return {"deleted": deleted}
