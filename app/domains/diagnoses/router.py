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
from app.core.exceptions import InferenceUnavailableError
from app.domains.action_plans.schemas import ActionPlanLevelResponse
from app.domains.auth.dto import UserDTO
from app.domains.chat.schemas import SemanticDiagnosisHit
from app.domains.diagnoses.schemas import (
    AnalyzedDiagnosisResponse,
    CreateDiagnosisRequest,
    DiagnosisActionPlanSchema,
    DiagnosisActionPlanSourceSchema,
    DiagnosisFilters,
    DiagnosisResponse,
    SetDiagnosisTalhaoRequest,
    TalhaoGroupResponse,
)
from app.domains.diagnoses.service import DiagnosisService
from app.domains.inference.service import InferenceService, resolve_allowed_model
from app.domains.subscriptions.features import PlanFeatures
from app.domains.usage.service import UsageService
from app.shared.enums import ActionPlanLevelEnum, FeatureTypeEnum, SeverityEnum
from app.shared.pagination import PaginatedResponse

if TYPE_CHECKING:
    from langgraph.store.postgres.aio import AsyncPostgresStore

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


def _to_action_plan(
    raw: dict[str, Any] | None, allowed_levels: list[ActionPlanLevelEnum]
) -> DiagnosisActionPlanSchema | None:
    """Converte o plano montado pelo ``compose_action_plan_node`` (TCC-105).

    ``None`` quando a doenca nao tem plano cadastrado — o node devolve
    ``levels`` e ``sources`` vazios nesse caso (ex.: folha saudavel).
    """
    if not raw or not (raw.get("levels") or raw.get("sources")):
        return None
    return DiagnosisActionPlanSchema(
        levels=[
            ActionPlanLevelResponse(
                level=ActionPlanLevelEnum(lvl["level"]), actions=list(lvl["actions"])
            )
            for lvl in raw.get("levels", [])
        ],
        allowed_levels=allowed_levels,
        sources=[
            DiagnosisActionPlanSourceSchema(name=s["name"], url=s.get("url"))
            for s in raw.get("sources", [])
        ],
    )


@router.post(
    "/analyze", response_model=list[AnalyzedDiagnosisResponse], status_code=200
)
async def analyze(
    response: Response,
    images: list[UploadFile] = File(...),
    crop_id: str = Form(default="soja"),
    model: str = Form(default="ensemble"),
    talhao_id: str | None = Form(default=None),
    current_user: UserDTO = Depends(require_quota_dual),
    auth_method: str = Depends(auth_method_dual),
    diagnosis_graph_factory: Callable[[str], Any] = Depends(get_diagnosis_graph_factory),
    diag_svc: DiagnosisService = Depends(get_diagnosis_service),
    usage_svc: UsageService = Depends(get_usage_service),
    usage_repo: "UsageRepository" = Depends(get_usage_repository),
    sub_repo: "SubscriptionRepository" = Depends(get_subscription_repository),
    plan_features: PlanFeatures = Depends(get_plan_features_dual),
) -> list[AnalyzedDiagnosisResponse]:
    """Endpoint REST direto pra diagnostico (sem chat).

    Auth dual: aceita JWT (web) OU ``X-API-Key`` (Enterprise via REST). Quota
    eh contada por INFERENCE (daily) com JWT, ou API (monthly) com API key.
    Quando autenticado via API key, response carrega headers
    ``X-RateLimit-{Limit,Remaining,Reset}`` (TCC-064).

    TCC-105: cada laudo vem com ``image_url`` assinada, ``sources`` e o
    ``action_plan`` filtrado pelo plano. Headers
    ``X-Model-Used``, ``X-Model-Downgraded`` e ``X-Images-Failed`` informam
    rebaixamento de modelo e fotos que ficaram de fora. Lote sem nenhuma foto
    classificada vira 503 e nao consome cota.
    """
    feature = (
        FeatureTypeEnum.API if auth_method == "api_key" else FeatureTypeEnum.INFERENCE
    )

    graph = diagnosis_graph_factory(crop_id)
    image_batch, image_ids = await _read_image_batch(images)
    effective_model, downgraded = resolve_allowed_model(
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
            "talhao_id": talhao_id or None,
        }
    )

    persisted_ids: list[str] = result.get("persisted_ids", [])
    failed: list[dict[str, Any]] = result.get("errors") or []

    # TCC-105: nenhuma imagem classificada nao e' "200 com lista vazia" — e' o
    # mesmo 503 de UX-001, e a cota nao e' consumida.
    if not persisted_ids and failed:
        raise InferenceUnavailableError(
            "Nenhuma imagem pôde ser analisada. "
            + " ".join(f"{f.get('image_name')}: {f.get('error')}" for f in failed)
        )

    # TCC-105: via service, nao via repositorio — e' ali que a storage key vira
    # URL assinada e que as fontes (``sources``) entram na resposta.
    diagnoses = await diag_svc.get_many(persisted_ids, current_user.id)
    # ``persist_node`` grava um laudo por predicao, na ordem; ``action_plans``
    # segue o mesmo index.
    action_plans: list[dict[str, Any]] = result.get("action_plans") or []
    allowed_levels = [
        ActionPlanLevelEnum(lvl) for lvl in plan_features.action_plan_levels
    ]
    responses = [
        AnalyzedDiagnosisResponse(
            **diagnosis.model_dump(),
            action_plan=_to_action_plan(
                action_plans[i] if i < len(action_plans) else None, allowed_levels
            ),
        )
        for i, diagnosis in enumerate(diagnoses)
    ]

    await usage_svc.record_usage(
        current_user.id,
        feature,
        {
            "crop_id": crop_id,
            "model": effective_model,
            "model_requested": model,
            "batch_size": len(images),
            "images_failed": len(failed),
            "auth_method": auth_method,
        },
    )

    # TCC-105: o cliente precisa saber quando o plano trocou o modelo pedido e
    # quantas fotos ficaram de fora do lote — o corpo continua sendo a lista de
    # laudos para nao quebrar integracoes existentes.
    response.headers["X-Model-Used"] = effective_model
    response.headers["X-Model-Downgraded"] = "true" if downgraded else "false"
    response.headers["X-Images-Failed"] = str(len(failed))

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
    talhao_id: str | None = Query(
        default=None,
        description='Id do talhao, ou "sem-talhao" para os laudos sem vinculo.',
    ),
    current_user: UserDTO = Depends(get_current_user),
    service: DiagnosisService = Depends(get_diagnosis_service),
) -> PaginatedResponse[DiagnosisResponse]:
    filters = DiagnosisFilters(
        page=page, limit=limit, severity=severity, search=search, talhao_id=talhao_id
    )
    return await service.list_for_user(current_user.id, filters)


@router.get("/por-talhao", response_model=list[TalhaoGroupResponse])
async def list_diagnoses_by_talhao(
    per_group: int = Query(default=3, ge=1, le=10),
    current_user: UserDTO = Depends(get_current_user),
    service: DiagnosisService = Depends(get_diagnosis_service),
) -> list[TalhaoGroupResponse]:
    """Historico agrupado por talhao (TCC-093).

    Um grupo por talhao do usuario (inclusive os ainda sem laudo) e, no fim,
    o grupo "Sem talhao" (``talhao_id`` null) quando houver laudos sem vinculo.
    Cada grupo traz o total e os ``per_group`` laudos mais recentes.
    """
    return await service.group_by_talhao(current_user.id, per_group)


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


@router.patch("/{diagnosis_id}/talhao", response_model=DiagnosisResponse)
async def set_diagnosis_talhao(
    diagnosis_id: str,
    body: SetDiagnosisTalhaoRequest,
    current_user: UserDTO = Depends(get_current_user),
    service: DiagnosisService = Depends(get_diagnosis_service),
) -> DiagnosisResponse:
    """Move o laudo para outro talhao (``talhao_id`` null = "Sem talhao")."""
    return await service.set_talhao(diagnosis_id, current_user.id, body.talhao_id)


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
