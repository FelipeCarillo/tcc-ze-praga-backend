"""Integration tests for /api/v1/chat router (TCC-010).

Após refatoração: router delega ao ChatService, que orquestra LangGraph.
Aqui mockamos o ChatService inteiro pra não depender do grafo nem do banco.
"""

import io
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.dependencies import (
    get_chat_service,
    get_current_user,
    get_usage_service,
    require_quota,
)
from app.domains.chat.schemas import ChatResponse
from app.main import app
from app.shared.enums import FeatureTypeEnum
from tests.conftest import make_diagnosis_dto, make_user_dto


@pytest.fixture
def fake_diagnosis_response():
    diag = make_diagnosis_dto()
    # ChatResponse.diagnosis espera DiagnosisResponse (pydantic) — convert.
    from app.domains.diagnoses.schemas import DiagnosisResponse, Top3PredictionSchema

    return DiagnosisResponse(
        id=diag.id,
        disease_name=diag.disease_name,
        disease_id=diag.disease_id,
        scientific_name=diag.scientific_name,
        confidence=diag.confidence,
        severity=diag.severity,
        description=diag.description,
        model_used=diag.model_used,
        image_url=diag.image_url,
        image_name=diag.image_name,
        created_at=diag.created_at,
        top3=[
            Top3PredictionSchema(
                rank=t.rank,
                disease_name=t.disease_name,
                disease_id=t.disease_id,
                scientific_name=t.scientific_name,
                confidence=t.confidence,
                severity=t.severity,
            )
            for t in diag.top3
        ],
    )


@pytest.fixture
def mock_chat_svc():
    svc = AsyncMock()
    svc.chat = AsyncMock()
    return svc


@pytest.fixture
def mock_usage_svc():
    svc = AsyncMock()
    svc.check_quota = AsyncMock()
    svc.record_usage = AsyncMock()
    return svc


@pytest.fixture
async def client_chat(mock_chat_svc, mock_usage_svc):
    app.dependency_overrides[require_quota(FeatureTypeEnum.CHAT)] = lambda: make_user_dto()
    app.dependency_overrides[get_chat_service] = lambda: mock_chat_svc
    app.dependency_overrides[get_usage_service] = lambda: mock_usage_svc
    app.dependency_overrides[get_current_user] = lambda: make_user_dto()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()


# ── Regressões + path básico ──────────────────────────────────────────────────


async def test_chat_falls_back_to_text_when_messages_is_not_json(client_chat, mock_chat_svc):
    """Regression for TCC-005: nested except tuple raised TypeError on JSONDecodeError."""
    mock_chat_svc.chat.return_value = ChatResponse(
        role="assistant",
        content="Olá! Como posso ajudar?",
        session_id="sess-1",
    )

    r = await client_chat.post(
        "/api/v1/chat",
        data={"messages": "ola", "model": "ensemble"},
    )

    assert r.status_code == 200
    body = r.json()
    assert body["role"] == "assistant"
    assert isinstance(body["content"], str)
    assert body["content"]

    # Service deve ter recebido o texto cru (não parseado como JSON)
    mock_chat_svc.chat.assert_awaited_once()
    call_kwargs = mock_chat_svc.chat.await_args.kwargs
    assert call_kwargs["message_text"] == "ola"
    assert call_kwargs["image_filename"] is None
    assert call_kwargs["model_id"] == "ensemble"


async def test_chat_parses_json_messages_payload(client_chat, mock_chat_svc):
    """Quando messages é JSON array, pega content do último turno."""
    mock_chat_svc.chat.return_value = ChatResponse(
        role="assistant",
        content="Resposta",
        session_id="sess-1",
    )

    payload = '[{"role":"user","content":"primeiro"},{"role":"user","content":"último"}]'
    r = await client_chat.post(
        "/api/v1/chat",
        data={"messages": payload, "model": "ensemble"},
    )

    assert r.status_code == 200
    assert mock_chat_svc.chat.await_args.kwargs["message_text"] == "último"


async def test_chat_text_only_no_image(client_chat, mock_chat_svc, mock_usage_svc):
    """Caminho sem imagem — service é chamado com image_filename=None."""
    mock_chat_svc.chat.return_value = ChatResponse(
        role="assistant",
        content="Para identificar ferrugem...",
        session_id="sess-1",
    )

    r = await client_chat.post(
        "/api/v1/chat",
        data={"messages": "me fala da ferrugem", "model": "vit"},
    )

    assert r.status_code == 200
    body = r.json()
    assert body["content"] == "Para identificar ferrugem..."
    assert body["session_id"] == "sess-1"
    assert body["diagnosis"] is None

    # Usage foi registrado
    mock_usage_svc.record_usage.assert_awaited_once()
    metadata = mock_usage_svc.record_usage.await_args.args[2]
    assert metadata == {"model": "vit", "has_image": False}


async def test_chat_with_image_returns_diagnosis(
    client_chat, mock_chat_svc, mock_usage_svc, fake_diagnosis_response
):
    """Upload de imagem — service recebe image_filename e devolve diagnosis."""
    mock_chat_svc.chat.return_value = ChatResponse(
        role="assistant",
        content="Detectei Ferrugem Asiática.",
        diagnosis=fake_diagnosis_response,
        session_id="sess-1",
    )

    files = {"image": ("folha.jpg", io.BytesIO(b"\x89PNG"), "image/jpeg")}
    data = {"messages": "analisa isso", "model": "ensemble"}

    r = await client_chat.post("/api/v1/chat", data=data, files=files)

    assert r.status_code == 200
    body = r.json()
    assert body["diagnosis"] is not None
    assert body["diagnosis"]["disease_id"] == "ferrugem-asiatica"

    call_kwargs = mock_chat_svc.chat.await_args.kwargs
    assert call_kwargs["image_filename"] == "folha.jpg"

    metadata = mock_usage_svc.record_usage.await_args.args[2]
    assert metadata == {"model": "ensemble", "has_image": True}


async def test_chat_passes_session_id_through(client_chat, mock_chat_svc):
    """Quando session_id é enviado, é repassado ao service."""
    mock_chat_svc.chat.return_value = ChatResponse(
        role="assistant",
        content="continuando...",
        session_id="existing-sess",
    )

    r = await client_chat.post(
        "/api/v1/chat",
        data={
            "messages": "oi",
            "model": "ensemble",
            "session_id": "existing-sess",
        },
    )

    assert r.status_code == 200
    assert mock_chat_svc.chat.await_args.kwargs["session_id"] == "existing-sess"
