"""Smoke — chat multimodal (TCC-083): campo `audio` + transcript + gate de visão.

Camada mockada: o gate de visão *real* (gpt-4o) e a transcrição *real* vivem nos
testes live (`test_agent_liveness.py`). Aqui garantimos o contrato HTTP do áudio
e que as tools de visão estão plugadas no chat.
"""

import io
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.dependencies import get_chat_service, get_transcription_service
from app.domains.chat.schemas import ChatResponse
from app.main import app
from tests.smoke.conftest import bypass_auth_overrides


@pytest.fixture
async def client_multimodal():
    chat_svc = AsyncMock()
    chat_svc.chat = AsyncMock(
        return_value=ChatResponse(role="assistant", content="ok", session_id="s1")
    )

    async def _stream(**kwargs):
        yield {"event": "token", "data": "oi"}
        yield {"event": "done", "data": "s1"}

    chat_svc.chat_stream = lambda **kwargs: _stream(**kwargs)

    transcription = AsyncMock()
    transcription.transcribe = AsyncMock(return_value="quero diagnosticar minha soja")

    bypass_auth_overrides()
    app.dependency_overrides[get_chat_service] = lambda: chat_svc
    app.dependency_overrides[get_transcription_service] = lambda: transcription
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac, chat_svc, transcription
    app.dependency_overrides.clear()


async def test_chat_accepts_audio_and_returns_transcript(client_multimodal):
    ac, chat_svc, transcription = client_multimodal
    files = {"audio": ("voice.webm", io.BytesIO(b"RIFFfakeaudio"), "audio/webm")}
    r = await ac.post("/api/v1/chat", data={"messages": "[]", "model": "ensemble"}, files=files)

    assert r.status_code == 200
    body = r.json()
    assert body["transcript"] == "quero diagnosticar minha soja"
    transcription.transcribe.assert_awaited_once()
    # O texto transcrito virou o message_text enviado ao agente.
    assert chat_svc.chat.await_args.kwargs["message_text"] == "quero diagnosticar minha soja"


async def test_chat_stream_emits_transcript_event(client_multimodal):
    ac, _, _ = client_multimodal
    files = {"audio": ("voice.webm", io.BytesIO(b"RIFFfakeaudio"), "audio/webm")}
    async with ac.stream(
        "POST", "/api/v1/chat/stream", data={"messages": "[]", "model": "ensemble"}, files=files
    ) as r:
        assert r.status_code == 200
        body = b""
        async for chunk in r.aiter_bytes():
            body += chunk

    text = body.decode("utf-8")
    assert "event: transcript" in text
    assert "quero diagnosticar minha soja" in text


async def test_chat_without_audio_has_null_transcript(client_multimodal):
    ac, _, transcription = client_multimodal
    r = await ac.post("/api/v1/chat", data={"messages": "oi", "model": "ensemble"})

    assert r.status_code == 200
    assert r.json()["transcript"] is None
    transcription.transcribe.assert_not_awaited()


def test_chat_tools_include_vision_gate():
    """O chat ao vivo monta inspect_image + analyze_image (gate de visão plugado)."""
    from app.domains.chat.agent import build_chat_tools

    tools = build_chat_tools(MagicMock(), MagicMock(), MagicMock())
    names = {t.name for t in tools}
    assert "inspect_image" in names
    assert "analyze_image" in names
    assert "get_action_plan" in names
