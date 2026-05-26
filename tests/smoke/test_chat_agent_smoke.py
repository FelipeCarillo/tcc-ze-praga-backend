"""Smoke tests for /api/v1/chat* and /api/v1/sessions/{id}/close (TCC-071).

Surface-level HTTP smoke — mockado. Verifica que cada rota está plugada,
exige auth e devolve schema mínimo válido. Asserts FINOS: nao duplica
a suíte de integração (tests/integration/test_chat_router.py).

Padrão de fixture-client:
  1. bypass_auth_overrides() — destrava get_current_user + require_quota(CHAT) + get_usage_service.
  2. sobrescreve get_chat_service com AsyncMock.
  3. AsyncClient(transport=ASGITransport(app=app)).
  4. limpa dependency_overrides no teardown.
"""

from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.dependencies import get_chat_service
from app.domains.chat.schemas import (
    ChatResponse,
    CloseSessionResponse,
    InterruptInfo,
    PendingInterrupt,
)
from app.main import app
from tests.conftest import make_user_dto
from tests.smoke.conftest import bypass_auth_overrides


# ── shared helpers ─────────────────────────────────────────────────────────────


def _make_chat_svc() -> AsyncMock:
    svc = AsyncMock()
    svc.chat = AsyncMock()
    svc.resume = AsyncMock()
    svc.list_pending_interrupts = AsyncMock()
    svc.close_session = AsyncMock()
    return svc


async def _async_gen(events):
    for e in events:
        yield e


# ── fixture ────────────────────────────────────────────────────────────────────


@pytest.fixture
async def chat_smoke_client():
    """Client com auth bypassado e ChatService mockado."""
    bypass_auth_overrides()
    mock_svc = _make_chat_svc()
    app.dependency_overrides[get_chat_service] = lambda: mock_svc
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac, mock_svc
    app.dependency_overrides.clear()


# ── POST /api/v1/chat ──────────────────────────────────────────────────────────


async def test_chat_returns_200_with_role(chat_smoke_client):
    """POST /chat — retorna 200 com campo role."""
    client, mock_svc = chat_smoke_client
    mock_svc.chat.return_value = ChatResponse(
        role="assistant",
        content="Resposta do agente.",
        session_id="sess-1",
    )

    r = await client.post(
        "/api/v1/chat",
        data={"messages": "oi", "model": "ensemble"},
    )

    assert r.status_code == 200
    assert r.json()["role"] == "assistant"


# ── POST /api/v1/chat/stream (SSE) ────────────────────────────────────────────


async def test_chat_stream_content_type_and_events(chat_smoke_client):
    """POST /chat/stream — content-type text/event-stream e emite event:/data:."""
    client, mock_svc = chat_smoke_client
    mock_svc.chat_stream = lambda **kwargs: _async_gen(
        [
            {"event": "token", "data": "Olá "},
            {"event": "done", "data": "sess-1"},
        ]
    )

    async with client.stream(
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
    assert "data:" in text


# ── POST /api/v1/chat/resume ──────────────────────────────────────────────────


async def test_resume_returns_200_with_session_id(chat_smoke_client):
    """POST /chat/resume — retorna 200 com session_id."""
    client, mock_svc = chat_smoke_client
    mock_svc.resume.return_value = ChatResponse(
        role="assistant",
        content="Continuando.",
        session_id="sess-1",
    )

    r = await client.post(
        "/api/v1/chat/resume",
        json={"thread_id": "sess-1", "response": "soja"},
    )

    assert r.status_code == 200
    assert r.json()["session_id"] == "sess-1"


# ── POST /api/v1/chat/resume/stream (SSE) ────────────────────────────────────


async def test_resume_stream_content_type_and_events(chat_smoke_client):
    """POST /chat/resume/stream — content-type SSE e emite event:/data:."""
    client, mock_svc = chat_smoke_client
    mock_svc.resume_stream = lambda **kwargs: _async_gen(
        [
            {"event": "token", "data": "ok"},
            {"event": "done", "data": "sess-1"},
        ]
    )

    async with client.stream(
        "POST",
        "/api/v1/chat/resume/stream",
        json={"thread_id": "sess-1", "response": "milho"},
    ) as r:
        assert r.status_code == 200
        assert "text/event-stream" in r.headers["content-type"]
        lines = []
        async for line in r.aiter_lines():
            lines.append(line)

    text = "\n".join(lines)
    assert "event: token" in text
    assert "data:" in text


# ── GET /api/v1/chat/interrupts ───────────────────────────────────────────────


async def test_list_interrupts_returns_200_list(chat_smoke_client):
    """GET /chat/interrupts — retorna 200 e lista (pode ser vazia)."""
    client, mock_svc = chat_smoke_client
    mock_svc.list_pending_interrupts.return_value = [
        PendingInterrupt(
            session_id="sess-A",
            interrupt=InterruptInfo(
                kind="ask_user",
                question="Qual cultivo?",
                response_kind="choice",
                options=["soja", "milho"],
            ),
        )
    ]

    r = await client.get("/api/v1/chat/interrupts")

    assert r.status_code == 200
    body = r.json()
    assert isinstance(body, list)
    assert body[0]["session_id"] == "sess-A"


# ── POST /api/v1/sessions/{session_id}/close ─────────────────────────────────


async def test_close_session_returns_200_with_session_id(chat_smoke_client):
    """POST /sessions/{id}/close — retorna 200 com session_id."""
    client, mock_svc = chat_smoke_client
    mock_svc.close_session.return_value = CloseSessionResponse(
        session_id="sess-1",
        summary_text="Resumo da sessão.",
    )

    r = await client.post("/api/v1/sessions/sess-1/close")

    assert r.status_code == 200
    assert r.json()["session_id"] == "sess-1"


# ── 401 sem auth ──────────────────────────────────────────────────────────────


async def test_chat_requires_auth(smoke_client):
    """POST /chat sem token — 401."""
    r = await smoke_client.post(
        "/api/v1/chat",
        data={"messages": "oi", "model": "ensemble"},
    )
    assert r.status_code == 401
