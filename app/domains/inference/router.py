from fastapi import APIRouter, Depends, File, Form, UploadFile

from app.core.dependencies import (
    get_diagnosis_service,
    get_inference_service,
    get_usage_service,
    require_quota,
)
from app.domains.auth.dto import UserDTO
from app.domains.diagnoses.schemas import CreateDiagnosisRequest, DiagnosisResponse
from app.domains.diagnoses.service import DiagnosisService
from app.domains.inference.service import InferenceService
from app.domains.usage.service import UsageService
from app.shared.enums import FeatureTypeEnum, ModelEnum

router = APIRouter(prefix="/inference", tags=["Inference"])


@router.post("", response_model=DiagnosisResponse, status_code=201)
async def run_inference(
    image: UploadFile = File(...),
    model: str = Form(default=ModelEnum.ENSEMBLE),
    current_user: UserDTO = Depends(require_quota(FeatureTypeEnum.INFERENCE)),
    diagnosis_svc: DiagnosisService = Depends(get_diagnosis_service),
    usage_svc: UsageService = Depends(get_usage_service),
    inference_svc: InferenceService = Depends(get_inference_service),
) -> DiagnosisResponse:
    result = inference_svc.predict(model, image.filename or "imagem.jpg")

    body = CreateDiagnosisRequest(
        disease_name=result.disease_name,
        disease_id=result.disease_id,
        scientific_name=result.scientific_name,
        confidence=result.confidence,
        severity=result.severity,
        description=result.description,
        model_used=result.model_id,
        image_url=None,
        image_name=result.image_name,
        top3=result.top3,
    )

    crop_uuid = (
        inference_svc.disease_catalog[0].crop_id
        if inference_svc.disease_catalog
        else ""
    )
    diagnosis = await diagnosis_svc.create(current_user.id, body, crop_id=crop_uuid)

    await usage_svc.record_usage(
        current_user.id,
        FeatureTypeEnum.INFERENCE,
        {"disease_id": result.disease_id, "model": model},
    )

    return diagnosis
