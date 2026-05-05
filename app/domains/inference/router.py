import random
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, File, Form, UploadFile

from app.core.dependencies import (
    get_diagnosis_service,
    get_usage_service,
    require_quota,
)
from app.domains.auth.dto import UserDTO
from app.domains.diagnoses.schemas import (
    CreateDiagnosisRequest,
    DiagnosisResponse,
    Top3PredictionSchema,
)
from app.domains.diagnoses.service import DiagnosisService
from app.domains.usage.service import UsageService
from app.shared.enums import FeatureTypeEnum, ModelEnum, SeverityEnum

router = APIRouter(prefix="/inference", tags=["Inference"])

_DISEASES = [
    {
        "id": "ferrugem-asiatica",
        "name": "Ferrugem Asiática",
        "scientific_name": "Phakopsora pachyrhizi",
        "severity": SeverityEnum.ALTA,
        "description": (
            "Doença fúngica severa que causa lesões amareladas a marrom-escuras na face inferior "
            "das folhas. É a principal doença da soja no Brasil, podendo causar desfolha precoce "
            "e perdas de até 80% na produtividade."
        ),
    },
    {
        "id": "mancha-alvo",
        "name": "Mancha-Alvo",
        "scientific_name": "Corynespora cassiicola",
        "severity": SeverityEnum.MEDIA,
        "description": (
            "Doença fúngica caracterizada por lesões circulares com anéis concêntricos que lembram "
            "um alvo. Causa desfolha prematura e redução na produtividade, especialmente em "
            "cultivares suscetíveis."
        ),
    },
    {
        "id": "antracnose",
        "name": "Antracnose",
        "scientific_name": "Colletotrichum truncatum",
        "severity": SeverityEnum.MEDIA,
        "description": (
            "Doença fúngica que afeta hastes, vagens e sementes, causando lesões escuras e "
            "deprimidas. Pode causar morte de plântulas e apodrecimento de vagens, reduzindo "
            "a qualidade e quantidade dos grãos."
        ),
    },
    {
        "id": "cercosporiose",
        "name": "Cercosporiose",
        "scientific_name": "Cercospora kikuchii",
        "severity": SeverityEnum.BAIXA,
        "description": (
            "Doença fúngica que causa manchas púrpuras nas sementes (mancha púrpura) e lesões "
            "foliares. Reduz a qualidade das sementes e pode causar perdas moderadas na "
            "produtividade quando ocorre em condições favoráveis."
        ),
    },
    {
        "id": "mildio",
        "name": "Míldio",
        "scientific_name": "Peronospora manshurica",
        "severity": SeverityEnum.BAIXA,
        "description": (
            "Doença causada por oomiceto que produz lesões amareladas na face superior das folhas "
            "com esporulação cinza-esbranquiçada na face inferior. Geralmente causa perdas "
            "econômicas moderadas, exceto em condições de alta umidade."
        ),
    },
    {
        "id": "saudavel",
        "name": "Saudável",
        "scientific_name": None,
        "severity": SeverityEnum.NENHUMA,
        "description": (
            "A folha analisada não apresenta sinais visíveis de doenças fúngicas ou bacterianas. "
            "Continue monitorando regularmente para detecção precoce de possíveis infecções."
        ),
    },
]


def _generate_mock_result(model_id: str, image_name: str) -> dict:
    primary_idx = random.randint(0, len(_DISEASES) - 1)
    primary = _DISEASES[primary_idx]

    base_confidence = 0.70 + random.random() * 0.25
    primary_confidence = round(min(0.99, base_confidence), 3)

    remaining = [d for i, d in enumerate(_DISEASES) if i != primary_idx]
    random.shuffle(remaining)
    leftover = 1.0 - primary_confidence
    second_conf = round(random.uniform(0.005, leftover * 0.7), 3)
    third_conf = round(max(0.001, leftover - second_conf), 3)

    top3 = [
        Top3PredictionSchema(
            rank=1,
            disease_name=primary["name"],
            disease_id=primary["id"],
            scientific_name=primary["scientific_name"],
            confidence=primary_confidence,
            severity=primary["severity"],
        ),
        Top3PredictionSchema(
            rank=2,
            disease_name=remaining[0]["name"],
            disease_id=remaining[0]["id"],
            scientific_name=remaining[0]["scientific_name"],
            confidence=second_conf,
            severity=remaining[0]["severity"],
        ),
        Top3PredictionSchema(
            rank=3,
            disease_name=remaining[1]["name"],
            disease_id=remaining[1]["id"],
            scientific_name=remaining[1]["scientific_name"],
            confidence=third_conf,
            severity=remaining[1]["severity"],
        ),
    ]

    return {
        "disease": primary,
        "confidence": primary_confidence,
        "top3": top3,
        "model_id": model_id,
        "image_name": image_name,
    }


@router.post("", response_model=DiagnosisResponse, status_code=201)
async def run_inference(
    image: UploadFile = File(...),
    model: str = Form(default=ModelEnum.ENSEMBLE),
    current_user: UserDTO = Depends(require_quota(FeatureTypeEnum.INFERENCE)),
    diagnosis_svc: DiagnosisService = Depends(get_diagnosis_service),
    usage_svc: UsageService = Depends(get_usage_service),
) -> DiagnosisResponse:
    result = _generate_mock_result(model, image.filename or "imagem.jpg")
    disease = result["disease"]

    body = CreateDiagnosisRequest(
        disease_name=disease["name"],
        disease_id=disease["id"],
        scientific_name=disease["scientific_name"],
        confidence=result["confidence"],
        severity=disease["severity"],
        description=disease["description"],
        model_used=result["model_id"],
        image_url=None,
        image_name=result["image_name"],
        top3=result["top3"],
    )

    diagnosis = await diagnosis_svc.create(current_user.id, body)

    await usage_svc.record_usage(
        current_user.id,
        FeatureTypeEnum.INFERENCE,
        {"disease_id": disease["id"], "model": model},
    )

    return diagnosis
