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
    assert metadata == {"model": "vit", "has_image": False, "has_audio": False}


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
    assert metadata == {"model": "ensemble", "has_image": True, "has_audio": False}


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


# ── SSE streaming (TCC-011) ───────────────────────────────────────────────────


async def _async_gen(events):
    for e in events:
        yield e


async def test_chat_stream_emits_token_and_done_events(client_chat, mock_chat_svc):
    """SSE endpoint streama tokens e marca done no final."""
    mock_chat_svc.chat_stream = lambda **kwargs: _async_gen(
        [
            {"event": "token", "data": "Hello "},
            {"event": "token", "data": "world"},
            {"event": "done", "data": "sess-1"},
        ]
    )

    async with client_chat.stream(
        "POST",
        "/api/v1/chat/stream",
        data={"messages": "oi", "model": "ensemble"},
    ) as r:
        assert r.status_code == 200
        assert "text/event-stream" in r.headers["content-type"]
        lines = []
        async for line in r.aiter_lines():
            lines.append(line)

    text = "\n".join(lines)
    assert "event: token" in text
    assert "data: Hello " in text
    assert "data: world" in text
    assert "event: done" in text


async def test_chat_stream_serializes_dict_data(client_chat, mock_chat_svc):
    """Quando data é dict (ex: diagnosis), serializa pra JSON-string."""
    mock_chat_svc.chat_stream = lambda **kwargs: _async_gen(
        [
            {
                "event": "diagnosis",
                "data": {"id": "diag-1", "disease": "Ferrugem"},
            },
            {"event": "done", "data": "sess-1"},
        ]
    )

    async with client_chat.stream(
        "POST",
        "/api/v1/chat/stream",
        data={"messages": "analisa", "model": "ensemble"},
    ) as r:
        body = b""
        async for chunk in r.aiter_bytes():
            body += chunk

    text = body.decode("utf-8")
    assert "event: diagnosis" in text
    # JSON-string deve ter sido emitido
    assert "diag-1" in text
    assert "Ferrugem" in text


async def test_chat_stream_records_usage_with_streaming_flag(
    client_chat, mock_chat_svc, mock_usage_svc
):
    mock_chat_svc.chat_stream = lambda **kwargs: _async_gen([{"event": "done", "data": "sess-1"}])

    async with client_chat.stream(
        "POST",
        "/api/v1/chat/stream",
        data={"messages": "oi", "model": "vit"},
    ) as r:
        async for _ in r.aiter_bytes():
            pass

    mock_usage_svc.record_usage.assert_awaited()
    metadata = mock_usage_svc.record_usage.await_args.args[2]
    assert metadata == {"model": "vit", "has_image": False, "has_audio": False, "streaming": True}


async def test_chat_stream_passes_image_filename(client_chat, mock_chat_svc):
    captured = {}

    async def _capturing_stream(**kwargs):
        captured.update(kwargs)
        yield {"event": "done", "data": "sess-1"}

    mock_chat_svc.chat_stream = _capturing_stream

    files = {"image": ("folha.jpg", io.BytesIO(b"\x89PNG"), "image/jpeg")}

    async with client_chat.stream(
        "POST",
        "/api/v1/chat/stream",
        data={"messages": "analisa", "model": "ensemble"},
        files=files,
    ) as r:
        async for _ in r.aiter_bytes():
            pass

    assert captured["image_filename"] == "folha.jpg"
    assert captured["message_text"] == "analisa"
    assert captured["model_id"] == "ensemble"


# ── HITL: /chat/resume + /chat/interrupts (TCC-058) ───────────────────────────


async def test_resume_invokes_service_and_returns_chat_response(
    client_chat, mock_chat_svc
):
    """POST /chat/resume passa thread_id+response e devolve ChatResponse."""
    mock_chat_svc.resume = AsyncMock(
        return_value=ChatResponse(
            role="assistant",
            content="Continuando com soja confirmada.",
            session_id="sess-1",
        )
    )

    r = await client_chat.post(
        "/api/v1/chat/resume",
        json={"thread_id": "sess-1", "response": "soja"},
    )

    assert r.status_code == 200
    body = r.json()
    assert body["content"] == "Continuando com soja confirmada."
    assert body["session_id"] == "sess-1"

    mock_chat_svc.resume.assert_awaited_once()
    kwargs = mock_chat_svc.resume.await_args.kwargs
    assert kwargs["thread_id"] == "sess-1"
    assert kwargs["response"] == "soja"


async def test_resume_returns_chained_interrupt(client_chat, mock_chat_svc):
    """Quando o agente dispara outra pergunta apos resume, interrupt fica no body."""
    from app.domains.chat.schemas import InterruptInfo

    mock_chat_svc.resume = AsyncMock(
        return_value=ChatResponse(
            role="assistant",
            content="",
            session_id="sess-1",
            interrupt=InterruptInfo(
                kind="ask_user",
                question="Confirma o plano de campo?",
                response_kind="boolean",
            ),
        )
    )

    r = await client_chat.post(
        "/api/v1/chat/resume",
        json={"thread_id": "sess-1", "response": "soja"},
    )

    assert r.status_code == 200
    body = r.json()
    assert body["interrupt"]["question"] == "Confirma o plano de campo?"
    assert body["interrupt"]["response_kind"] == "boolean"


async def test_resume_stream_passes_payload(client_chat, mock_chat_svc):
    captured = {}

    async def _stream(**kwargs):
        captured.update(kwargs)
        yield {"event": "token", "data": "ok"}
        yield {"event": "done", "data": "sess-1"}

    mock_chat_svc.resume_stream = _stream

    async with client_chat.stream(
        "POST",
        "/api/v1/chat/resume/stream",
        json={"thread_id": "sess-1", "response": "milho"},
    ) as r:
        assert r.status_code == 200
        body = b""
        async for chunk in r.aiter_bytes():
            body += chunk

    text = body.decode("utf-8")
    assert "event: token" in text
    assert "event: done" in text
    assert captured["thread_id"] == "sess-1"
    assert captured["response"] == "milho"


async def test_resume_stream_serializes_interrupt_event(client_chat, mock_chat_svc):
    """Quando resume_stream emite interrupt (dict), router serializa pra JSON."""

    async def _stream(**kwargs):
        yield {
            "event": "interrupt",
            "data": {
                "kind": "ask_user",
                "question": "Top-2 proximos — qual?",
                "response_kind": "choice",
                "options": ["ferrugem", "mancha-alvo"],
            },
        }
        yield {"event": "done", "data": "sess-1"}

    mock_chat_svc.resume_stream = _stream

    async with client_chat.stream(
        "POST",
        "/api/v1/chat/resume/stream",
        json={"thread_id": "sess-1", "response": "x"},
    ) as r:
        body = b""
        async for chunk in r.aiter_bytes():
            body += chunk

    text = body.decode("utf-8")
    assert "event: interrupt" in text
    assert "ferrugem" in text
    assert "mancha-alvo" in text


async def test_list_interrupts_returns_pending_list(client_chat, mock_chat_svc):
    from app.domains.chat.schemas import InterruptInfo, PendingInterrupt

    mock_chat_svc.list_pending_interrupts = AsyncMock(
        return_value=[
            PendingInterrupt(
                session_id="sess-A",
                interrupt=InterruptInfo(
                    kind="ask_user",
                    question="Qual cultivo?",
                    response_kind="choice",
                    options=["soja", "milho"],
                ),
                created_at="2026-05-22T12:00:00Z",
            ),
            PendingInterrupt(
                session_id="sess-B",
                interrupt=InterruptInfo(
                    kind="ask_user",
                    question="Confirma?",
                    response_kind="boolean",
                ),
            ),
        ]
    )

    r = await client_chat.get("/api/v1/chat/interrupts")

    assert r.status_code == 200
    body = r.json()
    assert len(body) == 2
    assert body[0]["session_id"] == "sess-A"
    assert body[0]["interrupt"]["options"] == ["soja", "milho"]
    assert body[1]["session_id"] == "sess-B"


async def test_list_interrupts_empty(client_chat, mock_chat_svc):
    mock_chat_svc.list_pending_interrupts = AsyncMock(return_value=[])

    r = await client_chat.get("/api/v1/chat/interrupts")

    assert r.status_code == 200
    assert r.json() == []
