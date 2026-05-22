from fastapi import APIRouter, Depends, File, Form, Query, UploadFile

from app.core.dependencies import (
    get_current_user,
    get_diagnosis_graph_factory,
    get_diagnosis_repository,
    get_diagnosis_service,
    get_usage_service,
    require_quota,
)
from app.core.exceptions import NotFoundError
from app.domains.auth.dto import UserDTO
from app.domains.diagnoses.schemas import (
    CreateDiagnosisRequest,
    DiagnosisFilters,
    DiagnosisResponse,
    Top3PredictionSchema,
)
from app.domains.diagnoses.service import DiagnosisService
from app.domains.usage.service import UsageService
from app.shared.enums import FeatureTypeEnum, SeverityEnum
from app.shared.pagination import PaginatedResponse

router = APIRouter(prefix="/diagnoses", tags=["Diagnoses"])


@router.post("/analyze", response_model=list[DiagnosisResponse], status_code=200)
async def analyze(
    images: list[UploadFile] = File(...),
    crop_id: str = Form(default="soja"),
    model: str = Form(default="ensemble"),
    current_user: UserDTO = Depends(require_quota(FeatureTypeEnum.INFERENCE)),
    diagnosis_graph_factory=Depends(get_diagnosis_graph_factory),
    diag_repo=Depends(get_diagnosis_repository),
    usage_svc: UsageService = Depends(get_usage_service),
) -> list[DiagnosisResponse]:
    """Endpoint REST direto pra diagnostico (sem chat). Tier API/Enterprise (Sprint A3).

    Invoca o sub-grafo ``diagnosis_graph[crop_id]`` com o batch de imagens e
    retorna a lista de ``DiagnosisResponse`` persistida. ``image_name`` no DB
    pega o ``filename`` original; bytes nao sao persistidos aqui — uploads
    devem usar ``POST /api/v1/uploads`` se persistencia for necessaria.
    """
    graph = diagnosis_graph_factory(crop_id)
    image_ids = [img.filename or f"image-{i}" for i, img in enumerate(images)]

    result = await graph.ainvoke(
        {
            "user_id": current_user.id,
            "crop_id": crop_id,
            "image_batch": [],
            "image_ids": image_ids,
            "model_id": model,
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
        FeatureTypeEnum.INFERENCE,
        {"crop_id": crop_id, "model": model, "batch_size": len(images)},
    )
    return responses


@router.post("", response_model=DiagnosisResponse, status_code=201)
async def create_diagnosis(
    body: CreateDiagnosisRequest,
    current_user: UserDTO = Depends(require_quota(FeatureTypeEnum.INFERENCE)),
    service: DiagnosisService = Depends(get_diagnosis_service),
    usage_svc: UsageService = Depends(get_usage_service),
) -> DiagnosisResponse:
    result = await service.create(current_user.id, body)
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
) -> dict:
    if not confirm:
        return {"detail": "Pass ?confirm=true to delete all diagnoses"}
    deleted = await service.clear_all(current_user.id)
    return {"deleted": deleted}
