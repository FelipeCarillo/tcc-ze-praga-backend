"""Historico de conversas — leitura das sessoes persistidas.

O backend gravava ``chat_sessions`` e ``chat_messages`` desde sempre, mas nao
expunha nenhuma rota de leitura: o chat recomecava do zero a cada reload e o
usuario nao tinha como voltar numa conversa.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

from app.domains.chat.dto import ChatMessageDTO, ChatSessionDTO
from app.domains.chat.service import ChatService

NOW = datetime(2026, 8, 24, 12, 0, tzinfo=UTC)


def _session(sess_id: str = "s1", summary: str | None = None) -> ChatSessionDTO:
    return ChatSessionDTO(
        id=sess_id,
        user_id="user-1",
        title=None,
        created_at=NOW,
        updated_at=NOW,
        summary_text=summary,
    )


def _message(msg_id: str, role: str, content: str) -> ChatMessageDTO:
    return ChatMessageDTO(
        id=msg_id,
        session_id="s1",
        role=role,
        content=content,
        diagnosis_id=None,
        created_at=NOW,
        metadata=None,
    )


def _svc(session_repo=None, message_repo=None) -> ChatService:
    return ChatService(
        session_repo=session_repo or AsyncMock(),
        message_repo=message_repo or AsyncMock(),
        inference_svc=MagicMock(),
        action_plan_svc=AsyncMock(),
        diagnosis_svc=AsyncMock(),
    )


async def test_lista_sessoes_com_previa_e_contagem() -> None:
    session_repo = AsyncMock()
    session_repo.list_with_preview = AsyncMock(
        return_value=[
            (_session("s1", summary="Falamos de ferrugem."), 6, "olha essa folha"),
            (_session("s2"), 2, "e essa aqui?"),
        ]
    )
    svc = _svc(session_repo)

    sessions = await svc.list_sessions("user-1")

    assert [s.id for s in sessions] == ["s1", "s2"]
    assert sessions[0].preview == "olha essa folha"
    assert sessions[0].message_count == 6
    assert sessions[0].summary_text == "Falamos de ferrugem."
    session_repo.list_with_preview.assert_awaited_once_with("user-1", limit=50)


async def test_mensagens_da_sessao_em_ordem() -> None:
    session_repo = AsyncMock()
    session_repo.get_by_id = AsyncMock(return_value=_session())
    message_repo = AsyncMock()
    message_repo.list_by_session = AsyncMock(
        return_value=[
            _message("m1", "user", "olha essa folha"),
            _message("m2", "assistant", "Isso e ferrugem."),
        ]
    )
    svc = _svc(session_repo, message_repo)

    messages = await svc.get_session_messages("user-1", "s1")

    assert [m.role for m in messages] == ["user", "assistant"]
    assert messages[1].content == "Isso e ferrugem."


async def test_sessao_de_outro_usuario_devolve_vazio() -> None:
    """``get_by_id`` filtra por user_id; sem sessao, nao le mensagem nenhuma."""
    session_repo = AsyncMock()
    session_repo.get_by_id = AsyncMock(return_value=None)
    message_repo = AsyncMock()
    svc = _svc(session_repo, message_repo)

    messages = await svc.get_session_messages("user-1", "s-de-outro")

    assert messages == []
    message_repo.list_by_session.assert_not_awaited()
