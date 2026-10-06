"""TCC-097 — listar conversas não depende da inferência (modelos ONNX).

Antes, ``GET /sessions`` montava o ChatService inteiro e quebrava (500) onde os
modelos não estavam carregados. Aqui a inferência explode se for montada.
"""

from unittest.mock import AsyncMock, MagicMock

from httpx import ASGITransport, AsyncClient

from app.core.dependencies import (
    get_chat_message_repository,
    get_chat_session_repository,
    get_current_user,
    get_inference_service,
    get_upload_service,
)
from app.domains.chat.dto import SessionPreviewDTO
from app.main import app
from tests.conftest import NOW, make_user_dto
from tests.unit.chat.test_service import _fake_session
from tests.unit.chat.test_service_sessions import _message


def _sem_onnx():
    raise RuntimeError("modelos ONNX nao carregados")


async def test_lista_e_mensagens_sem_inferencia():
    sessions = AsyncMock()
    sessions.list_with_preview = AsyncMock(
        return_value=[SessionPreviewDTO(_fake_session("s1"), 2, "oi", last_reply="Olá!")]
    )
    sessions.get_by_id = AsyncMock(return_value=_fake_session("s1"))
    messages = AsyncMock()
    messages.list_by_session = AsyncMock(return_value=[_message("m1", "user", "oi")])
    upload = MagicMock()
    upload.signed_urls = MagicMock(return_value={})
    app.dependency_overrides.update(
        {
            get_current_user: lambda: make_user_dto(),
            get_chat_session_repository: lambda: sessions,
            get_chat_message_repository: lambda: messages,
            get_upload_service: lambda: upload,
            get_inference_service: _sem_onnx,
        }
    )
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            r = await ac.get("/api/v1/sessions")
            assert r.status_code == 200
            assert r.json()[0]["last_reply"] == "Olá!"
            r = await ac.get("/api/v1/sessions/s1/messages")
            assert r.status_code == 200 and r.json()[0]["content"] == "oi"
    finally:
        app.dependency_overrides.clear()
    assert NOW
