"""TCC-097 — lista de conversas contra um Postgres real.

Cobre as subqueries do card de "Conversas": ultima resposta do Ze, contagem de
laudos distintos e a foto/talhao do laudo mais recente. Pulado sem banco
(``ZP_TEST_PG_URL``), como ``test_diagnoses_talhao_db.py``.
"""

import os
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.domains.chat.repository import ChatSessionRepository
from app.models.chat_message import ChatMessage
from app.models.chat_session import ChatSession
from app.models.crop import Crop
from app.models.diagnosis import Diagnosis
from app.models.fazenda import Fazenda
from app.models.talhao import Talhao
from app.models.user import User
from app.shared.enums import SeverityEnum

PG_URL = os.environ.get("ZP_TEST_PG_URL")
pytestmark = pytest.mark.skipif(not PG_URL, reason="sem ZP_TEST_PG_URL (Postgres real)")


@pytest.fixture
async def session():
    engine = create_async_engine(PG_URL)  # type: ignore[arg-type]
    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as s:
        yield s
    await engine.dispose()


def _diag(user, crop, talhao, image):
    return Diagnosis(
        user_id=user,
        crop_id=crop,
        disease_name="Ferrugem",
        disease_id="ferrugem-asiatica",
        confidence=0.9,
        severity=SeverityEnum.ALTA,
        model_used="ensemble",
        talhao_id=talhao,
        image_url=image,
    )


async def test_card_da_conversa(session):
    tag = uuid.uuid4().hex[:8]
    crop = Crop(slug=f"soja-{tag}", name_pt="Soja")
    user = User(email=f"s-{tag}@t.test", password_hash="x")
    session.add_all([crop, user])
    await session.flush()
    fazenda = Fazenda(user_id=user.id, nome="Boa Vista")
    session.add(fazenda)
    await session.flush()
    sede = Talhao(user_id=user.id, fazenda_id=fazenda.id, nome="Sede")
    session.add(sede)
    await session.flush()
    antigo = _diag(user.id, crop.id, None, "k/antigo.jpg")
    novo = _diag(user.id, crop.id, sede.id, "k/novo.jpg")
    session.add_all([antigo, novo])
    conversa = ChatSession(user_id=user.id)
    vazia = ChatSession(user_id=user.id)
    pergunta = ChatSession(user_id=user.id)
    session.add_all([conversa, vazia, pergunta])
    await session.flush()

    t0 = datetime.now(UTC) - timedelta(hours=1)
    msgs = [
        ChatMessage(session_id=conversa.id, role="user", content="olha essa folha", created_at=t0),
        ChatMessage(session_id=conversa.id, role="assistant", content="Ferrugem.",
                    diagnosis_id=antigo.id, created_at=t0 + timedelta(minutes=1)),
        ChatMessage(session_id=conversa.id, role="user", content="e essa?",
                    created_at=t0 + timedelta(minutes=2)),
        ChatMessage(session_id=conversa.id, role="assistant", content="De novo ferrugem, na Sede.",
                    diagnosis_id=novo.id, created_at=t0 + timedelta(minutes=3)),
        ChatMessage(session_id=pergunta.id, role="user", content="o que e mildio?", created_at=t0),
        ChatMessage(session_id=pergunta.id, role="assistant", content="Um fungo.",
                    created_at=t0 + timedelta(minutes=1)),
    ]
    session.add_all(msgs)
    await session.commit()

    rows = await ChatSessionRepository(session).list_with_preview(user.id)
    by_id = {r.session.id: r for r in rows}

    assert vazia.id not in by_id  # conversa sem mensagem nao aparece
    card = by_id[conversa.id]
    assert card.preview == "olha essa folha"
    assert card.last_reply == "De novo ferrugem, na Sede."
    assert card.diagnosis_count == 2
    assert card.image_key == "k/novo.jpg"
    assert card.talhao_nome == "Sede"
    so_texto = by_id[pergunta.id]
    assert so_texto.diagnosis_count == 0
    assert so_texto.image_key is None and so_texto.talhao_nome is None
