import logging

from fastapi import APIRouter, Depends, File, Form, UploadFile

from app.core.dependencies import (
    get_diagnosis_service,
    get_inference_service,
    get_plan_features,
    get_upload_service,
    get_usage_service,
    require_quota,
)
from app.domains.auth.dto import UserDTO
from app.domains.diagnoses.schemas import CreateDiagnosisRequest, DiagnosisResponse
from app.domains.diagnoses.service import DiagnosisService
from app.domains.inference.service import InferenceService, resolve_allowed_model
from app.domains.subscriptions.features import PlanFeatures
from app.domains.uploads.service import UploadService
from app.domains.usage.service import UsageService
from app.shared.enums import FeatureTypeEnum, ModelEnum

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/inference", tags=["Inference"])


@router.post("", response_model=DiagnosisResponse, status_code=201)
async def run_inference(
    image: UploadFile = File(...),
    model: str = Form(default=ModelEnum.ENSEMBLE),
    current_user: UserDTO = Depends(require_quota(FeatureTypeEnum.INFERENCE)),
    diagnosis_svc: DiagnosisService = Depends(get_diagnosis_service),
    usage_svc: UsageService = Depends(get_usage_service),
    inference_svc: InferenceService = Depends(get_inference_service),
    plan_features: PlanFeatures = Depends(get_plan_features),
    upload_svc: UploadService = Depends(get_upload_service),
) -> DiagnosisResponse:
    image_bytes = await image.read()
    # O plano decide o modelo efetivo (TCC-051): pedido fora do tier cai no
    # melhor permitido em vez de 403.
    effective_model, _downgraded = resolve_allowed_model(
        model, plan_features.diagnosis_models
    )
    result = inference_svc.predict(
        effective_model, image.filename or "imagem.jpg", image_bytes=image_bytes
    )

    # Guarda a foto pra ela aparecer no histórico. Best-effort: falha de Storage
    # não pode custar o diagnóstico que o usuário já pagou em quota.
    storage_key: str | None = None
    try:
        row, _dedup = await upload_svc.upload(
            user_id=current_user.id,
            original_name=image.filename or "imagem.jpg",
            mime=image.content_type or "image/jpeg",
            data=image_bytes,
        )
        storage_key = row.storage_key
    except Exception:  # noqa: BLE001
        logger.exception("Falha ao subir imagem de /inference pro Storage")

    body = CreateDiagnosisRequest(
        disease_name=result.disease_name,
        disease_id=result.disease_id,
        scientific_name=result.scientific_name,
        confidence=result.confidence,
        severity=result.severity,
        description=result.description,
        model_used=result.model_id,
        image_url=storage_key,
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
        {
            "disease_id": result.disease_id,
            "model": effective_model,
            "model_requested": model,
        },
    )

    return diagnosis
