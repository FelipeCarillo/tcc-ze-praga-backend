"""Testes unitários dos repositórios de chat (sessions + messages)."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.domains.chat.dto import ChatMessageDTO, ChatSessionDTO
from app.domains.chat.repository import ChatMessageRepository, ChatSessionRepository

NOW = datetime(2026, 5, 22, 12, 0, 0, tzinfo=UTC)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_session_orm(
    id_: str = "sess-1",
    user_id: str = "user-1",
    title: str | None = None,
    summary_text: str | None = None,
):
    m = MagicMock()
    m.id = id_
    m.user_id = user_id
    m.title = title
    m.summary_text = summary_text
    m.created_at = NOW
    m.updated_at = NOW
    return m


def _make_message_orm(
    id_: str = "msg-1",
    session_id: str = "sess-1",
    role: str = "user",
    content: str = "ola",
    diagnosis_id: str | None = None,
    metadata: dict | None = None,
):
    m = MagicMock()
    m.id = id_
    m.session_id = session_id
    m.role = role
    m.content = content
    m.diagnosis_id = diagnosis_id
    m.created_at = NOW
    m.metadata_ = metadata
    return m


@pytest.fixture
def mock_db():
    db = AsyncMock()
    db.execute = AsyncMock(return_value=MagicMock())
    db.add = MagicMock()
    db.commit = AsyncMock()
    db.refresh = AsyncMock()
    return db


# ── ChatSessionRepository ─────────────────────────────────────────────────────


async def test_create_session_persists_and_returns_dto(mock_db):
    repo = ChatSessionRepository(mock_db)

    def _refresh(obj):
        obj.id = "new-sess"
        obj.user_id = "user-1"
        obj.title = "topo"
        obj.summary_text = None
        obj.created_at = NOW
        obj.updated_at = NOW

    mock_db.refresh.side_effect = _refresh

    dto = await repo.create("user-1", title="topo")

    assert isinstance(dto, ChatSessionDTO)
    assert dto.id == "new-sess"
    assert dto.user_id == "user-1"
    assert dto.title == "topo"
    mock_db.commit.assert_awaited_once()


async def test_get_by_id_returns_none_when_missing(mock_db):
    repo = ChatSessionRepository(mock_db)
    mock_db.execute.return_value.scalar_one_or_none.return_value = None
    assert await repo.get_by_id("missing") is None


async def test_get_by_id_with_user_filter_returns_dto(mock_db):
    repo = ChatSessionRepository(mock_db)
    mock_db.execute.return_value.scalar_one_or_none.return_value = _make_session_orm()
    dto = await repo.get_by_id("sess-1", user_id="user-1")
    assert dto is not None
    assert dto.id == "sess-1"


async def test_get_or_create_returns_existing_if_match(mock_db):
    repo = ChatSessionRepository(mock_db)
    mock_db.execute.return_value.scalar_one_or_none.return_value = _make_session_orm()

    dto = await repo.get_or_create_for_user("user-1", "sess-1")
    assert dto.id == "sess-1"
    # Não deve ter chamado commit (não criou nova)
    mock_db.commit.assert_not_awaited()


async def test_get_or_create_creates_when_session_id_missing(mock_db):
    repo = ChatSessionRepository(mock_db)
    mock_db.execute.return_value.scalar_one_or_none.return_value = None

    def _refresh(obj):
        obj.id = "fresh"
        obj.user_id = "user-1"
        obj.title = None
        obj.summary_text = None
        obj.created_at = NOW
        obj.updated_at = NOW

    mock_db.refresh.side_effect = _refresh

    dto = await repo.get_or_create_for_user("user-1", "non-existent")
    assert dto.id == "fresh"
    mock_db.commit.assert_awaited_once()


async def test_get_or_create_creates_when_session_id_none(mock_db):
    repo = ChatSessionRepository(mock_db)

    def _refresh(obj):
        obj.id = "fresh"
        obj.user_id = "user-1"
        obj.title = None
        obj.summary_text = None
        obj.created_at = NOW
        obj.updated_at = NOW

    mock_db.refresh.side_effect = _refresh

    dto = await repo.get_or_create_for_user("user-1", None)
    assert dto.id == "fresh"


async def test_update_summary_persists_and_returns_dto(mock_db):
    """TCC-048: update_summary atualiza summary_text e retorna DTO."""
    repo = ChatSessionRepository(mock_db)
    session = _make_session_orm()
    mock_db.execute.return_value.scalar_one_or_none.return_value = session

    def _refresh(obj):
        obj.summary_text = "Resumo final."

    mock_db.refresh.side_effect = _refresh

    dto = await repo.update_summary("sess-1", "user-1", "Resumo final.")

    assert dto is not None
    assert dto.summary_text == "Resumo final."
    assert session.summary_text == "Resumo final."
    mock_db.commit.assert_awaited_once()


async def test_update_summary_returns_none_when_session_missing(mock_db):
    repo = ChatSessionRepository(mock_db)
    mock_db.execute.return_value.scalar_one_or_none.return_value = None

    result = await repo.update_summary("missing", "user-1", "foo")

    assert result is None
    mock_db.commit.assert_not_awaited()


async def test_list_for_user_returns_dtos(mock_db):
    """TCC-058: list_for_user retorna sessoes ordenadas por updated_at desc."""
    repo = ChatSessionRepository(mock_db)
    s1 = _make_session_orm(id_="sess-A", user_id="user-1")
    s2 = _make_session_orm(id_="sess-B", user_id="user-1")
    mock_db.execute.return_value.scalars.return_value.all.return_value = [s1, s2]

    items = await repo.list_for_user("user-1")
    assert len(items) == 2
    assert items[0].id == "sess-A"
    assert items[1].id == "sess-B"


async def test_list_for_user_empty(mock_db):
    repo = ChatSessionRepository(mock_db)
    mock_db.execute.return_value.scalars.return_value.all.return_value = []
    assert await repo.list_for_user("user-1") == []


# ── ChatMessageRepository ─────────────────────────────────────────────────────


async def test_create_message_persists(mock_db):
    repo = ChatMessageRepository(mock_db)

    def _refresh(obj):
        obj.id = "msg-new"
        obj.session_id = "sess-1"
        obj.role = "user"
        obj.content = "ola"
        obj.diagnosis_id = None
        obj.created_at = NOW
        obj.metadata_ = None

    mock_db.refresh.side_effect = _refresh

    dto = await repo.create(session_id="sess-1", role="user", content="ola")
    assert isinstance(dto, ChatMessageDTO)
    assert dto.id == "msg-new"
    assert dto.content == "ola"
    mock_db.commit.assert_awaited_once()


async def test_create_message_with_diagnosis_and_metadata(mock_db):
    repo = ChatMessageRepository(mock_db)

    def _refresh(obj):
        obj.id = "msg-new"
        obj.session_id = "sess-1"
        obj.role = "assistant"
        obj.content = "Detectei..."
        obj.diagnosis_id = "diag-1"
        obj.created_at = NOW
        obj.metadata_ = {"image_filename": "f.jpg"}

    mock_db.refresh.side_effect = _refresh

    dto = await repo.create(
        session_id="sess-1",
        role="assistant",
        content="Detectei...",
        diagnosis_id="diag-1",
        metadata={"image_filename": "f.jpg"},
    )
    assert dto.diagnosis_id == "diag-1"
    assert dto.metadata == {"image_filename": "f.jpg"}


async def test_list_by_session_returns_ordered_messages(mock_db):
    repo = ChatMessageRepository(mock_db)
    m1 = _make_message_orm(id_="m1", role="user", content="a")
    m2 = _make_message_orm(id_="m2", role="assistant", content="b")
    mock_db.execute.return_value.scalars.return_value.all.return_value = [m1, m2]

    items = await repo.list_by_session("sess-1")
    assert len(items) == 2
    assert items[0].content == "a"
    assert items[1].content == "b"
