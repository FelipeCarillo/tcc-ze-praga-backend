"""Integration tests for /api/v1/inference router (TCC-012)."""

import io
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.dependencies import (
    get_current_user,
    get_diagnosis_service,
    get_inference_service,
    get_plan_features,
    get_usage_service,
    require_quota,
)
from app.domains.subscriptions.features import ENTERPRISE_FEATURES
from app.domains.diagnoses.schemas import DiagnosisResponse, Top3PredictionSchema
from app.domains.inference.schemas import InferenceResult
from app.main import app
from app.shared.enums import FeatureTypeEnum, SeverityEnum
from tests.conftest import make_diagnosis_dto, make_user_dto


@pytest.fixture
def fake_inference_result():
    return InferenceResult(
        disease_id="ferrugem-asiatica",
        disease_name="Ferrugem Asiática",
        scientific_name="Phakopsora pachyrhizi",
        severity=SeverityEnum.ALTA,
        description="Doença fúngica severa.",
        confidence=0.94,
        model_id="ensemble",
        image_name="folha.jpg",
        top3=[
            Top3PredictionSchema(
                rank=1,
                disease_name="Ferrugem Asiática",
                disease_id="ferrugem-asiatica",
                scientific_name="Phakopsora pachyrhizi",
                confidence=0.94,
                severity=SeverityEnum.ALTA,
            ),
            Top3PredictionSchema(
                rank=2,
                disease_name="Mancha-Alvo",
                disease_id="mancha-alvo",
                scientific_name="Corynespora cassiicola",
                confidence=0.04,
                severity=SeverityEnum.MEDIA,
            ),
            Top3PredictionSchema(
                rank=3,
                disease_name="Saudável",
                disease_id="saudavel",
                scientific_name=None,
                confidence=0.02,
                severity=SeverityEnum.NENHUMA,
            ),
        ],
    )


@pytest.fixture
def fake_diagnosis_response():
    d = make_diagnosis_dto()
    return DiagnosisResponse(
        id=d.id,
        disease_name=d.disease_name,
        disease_id=d.disease_id,
        scientific_name=d.scientific_name,
        confidence=d.confidence,
        severity=d.severity,
        description=d.description,
        model_used=d.model_used,
        image_url=d.image_url,
        image_name=d.image_name,
        created_at=d.created_at,
        top3=[
            Top3PredictionSchema(
                rank=t.rank,
                disease_name=t.disease_name,
                disease_id=t.disease_id,
                scientific_name=t.scientific_name,
                confidence=t.confidence,
                severity=t.severity,
            )
            for t in d.top3
        ],
    )


@pytest.fixture
def mock_inference_svc(fake_inference_result):
    svc = MagicMock()
    svc.predict = MagicMock(return_value=fake_inference_result)
    return svc


@pytest.fixture
def mock_diagnosis_svc(fake_diagnosis_response):
    svc = AsyncMock()
    svc.create = AsyncMock(return_value=fake_diagnosis_response)
    return svc


@pytest.fixture
def mock_usage_svc():
    svc = AsyncMock()
    svc.check_quota = AsyncMock()
    svc.record_usage = AsyncMock()
    return svc


@pytest.fixture
async def client_inference(mock_inference_svc, mock_diagnosis_svc, mock_usage_svc):
    app.dependency_overrides[require_quota(FeatureTypeEnum.INFERENCE)] = lambda: make_user_dto()
    app.dependency_overrides[get_inference_service] = lambda: mock_inference_svc
    app.dependency_overrides[get_diagnosis_service] = lambda: mock_diagnosis_svc
    app.dependency_overrides[get_usage_service] = lambda: mock_usage_svc
    app.dependency_overrides[get_current_user] = lambda: make_user_dto()
    # Enterprise: libera os 4 modelos, pro gate de plano nao trocar o pedido.
    app.dependency_overrides[get_plan_features] = lambda: ENTERPRISE_FEATURES
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()


async def test_run_inference_returns_diagnosis(
    client_inference, mock_inference_svc, mock_diagnosis_svc, mock_usage_svc
):
    """POST /api/v1/inference com imagem mocka inferência e persiste diagnose."""
    files = {"image": ("folha.jpg", io.BytesIO(b"\x89PNG"), "image/jpeg")}
    data = {"model": "ensemble"}

    r = await client_inference.post("/api/v1/inference", data=data, files=files)

    assert r.status_code == 201
    body = r.json()
    assert body["disease_id"] == "ferrugem-asiatica"
    assert body["disease_name"] == "Ferrugem Asiática"

    # Service chamado com filename + model + bytes da imagem (TCC-023)
    mock_inference_svc.predict.assert_called_once_with(
        "ensemble", "folha.jpg", image_bytes=b"\x89PNG"
    )

    # Diagnose persistido com payload do InferenceResult
    mock_diagnosis_svc.create.assert_awaited_once()
    create_args = mock_diagnosis_svc.create.await_args
    user_id, body_arg = create_args.args
    assert user_id == "user-uuid-1"
    assert body_arg.disease_id == "ferrugem-asiatica"
    assert body_arg.confidence == 0.94
    assert body_arg.model_used == "ensemble"
    assert len(body_arg.top3) == 3

    # Usage registrado
    mock_usage_svc.record_usage.assert_awaited_once()
    metadata = mock_usage_svc.record_usage.await_args.args[2]
    assert metadata == {
        "disease_id": "ferrugem-asiatica",
        "model": "ensemble",
        "model_requested": "ensemble",
    }


async def test_run_inference_with_alternate_model(client_inference, mock_inference_svc):
    """Endpoint aceita diferentes modelos e normaliza pro id canonico.

    O gate de plano (TCC-051) resolve o modelo efetivo antes de chamar o
    service, entao 'vit' chega em predict como 'vit_b16'.
    """
    files = {"image": ("imagem.png", io.BytesIO(b"\x89PNG"), "image/png")}
    data = {"model": "vit"}

    r = await client_inference.post("/api/v1/inference", data=data, files=files)

    assert r.status_code == 201
    mock_inference_svc.predict.assert_called_once_with(
        "vit_b16", "imagem.png", image_bytes=b"\x89PNG"
    )
