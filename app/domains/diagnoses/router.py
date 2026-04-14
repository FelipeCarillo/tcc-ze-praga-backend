from fastapi import APIRouter, Depends, Query

from app.core.dependencies import (
    get_current_user,
    get_diagnosis_service,
    get_usage_service,
    require_quota,
)
from app.domains.auth.dto import UserDTO
from app.domains.diagnoses.schemas import (
    CreateDiagnosisRequest,
    DiagnosisFilters,
    DiagnosisResponse,
)
from app.domains.diagnoses.service import DiagnosisService
from app.domains.usage.service import UsageService
from app.shared.enums import FeatureTypeEnum, SeverityEnum
from app.shared.pagination import PaginatedResponse

router = APIRouter(prefix="/diagnoses", tags=["Diagnoses"])


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
