"""TCC-098 — o talhão escolhido/criado pelo agente chega à resposta do chat."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage

import pytest

from app.domains.chat.service import ChatService
from tests.unit.chat.test_service import _drain, _fake_diagnosis_response, _fake_session


@pytest.fixture
def session_repo():
    repo = AsyncMock()
    repo.get_or_create_for_user.return_value = _fake_session()
    return repo


@pytest.fixture
def message_repo():
    repo = AsyncMock()
    repo.create.return_value = MagicMock()
    return repo


@pytest.fixture
def diagnosis_svc():
    return AsyncMock()


@pytest.fixture
def chat_service(session_repo, message_repo, diagnosis_svc):
    return ChatService(
        session_repo=session_repo,
        message_repo=message_repo,
        inference_svc=MagicMock(),
        action_plan_svc=AsyncMock(),
        diagnosis_svc=diagnosis_svc,
    )

SEL = {
    "id": "t-9",
    "nome": "Talhão 9",
    "apelido": "Rio",
    "hectares": 40.0,
    "data_semeadura": "2026-09-12",
    "fazenda_id": "f-1",
    "fazenda_nome": "Boa Vista",
    "created": True,
}


def _snapshot(values):
    snap = MagicMock()
    snap.tasks = []
    snap.values = values
    return snap


async def test_chat_sincrono_devolve_talhao(chat_service):
    graph = AsyncMock()
    graph.ainvoke = AsyncMock(
        return_value={"messages": [AIMessage(content="Criei o talhão.")], "talhao_selected": SEL}
    )
    with patch("app.domains.chat.service.build_graph", return_value=graph):
        resp = await chat_service.chat("u-1", None, "cria o talhao 9", None, None, None, model_id="ensemble")
    assert resp.talhao is not None and resp.talhao.created and resp.talhao.fazenda_nome == "Boa Vista"
    assert graph.ainvoke.await_args.args[0]["talhao_selected"] is None  # zera a cada turno


async def test_chat_sincrono_sem_talhao(chat_service):
    graph = AsyncMock()
    graph.ainvoke = AsyncMock(return_value={"messages": [AIMessage(content="oi")]})
    with patch("app.domains.chat.service.build_graph", return_value=graph):
        resp = await chat_service.chat("u-1", None, "oi", None, None, None, model_id="ensemble")
    assert resp.talhao is None


def test_talhao_invalido_no_estado_e_ignorado():
    assert ChatService._talhao_from({"talhao_selected": {"nome": "sem id"}}) is None
    assert ChatService._talhao_from({"talhao_selected": "x"}) is None
    assert ChatService._talhao_from(None) is None


async def test_stream_emite_talhao(chat_service):
    graph = MagicMock()

    async def _events(*a, **k):
        return
        yield  # pragma: no cover

    graph.astream_events = _events
    graph.aget_state = AsyncMock(return_value=_snapshot({"talhao_selected": SEL}))
    with patch("app.domains.chat.service.build_graph", return_value=graph):
        events = await _drain(
            chat_service.chat_stream(
                user_id="u-1", session_id=None, message_text="cria", image_bytes=None,
                image_mime=None, image_filename=None, model_id="ensemble",
            )
        )
    talhao = [e for e in events if e["event"] == "talhao"]
    assert len(talhao) == 1 and json.loads(talhao[0]["data"])["id"] == "t-9"


async def test_resume_traz_laudo_e_talhao_novos(chat_service, session_repo, message_repo, diagnosis_svc):
    """A pergunta do talhão veio antes da análise: o laudo nasce no resume."""
    session_repo.get_by_id.return_value = _fake_session("sess-1")
    diagnosis_svc.get_by_id = AsyncMock(return_value=_fake_diagnosis_response("diag-1"))
    graph = AsyncMock()
    graph.aget_state = AsyncMock(return_value=_snapshot({"diagnoses_in_turn": [], "talhao_selected": None}))
    graph.ainvoke = AsyncMock(
        return_value={
            "messages": [AIMessage(content="Ferrugem no Talhão 9.")],
            "diagnoses_in_turn": ["diag-1"],
            "talhao_selected": SEL,
        }
    )
    with patch("app.domains.chat.service.build_graph", return_value=graph):
        resp = await chat_service.resume("u-1", "sess-1", "Novo talhão")
    assert resp.diagnosis is not None and resp.diagnosis.id == "diag-1"
    assert resp.talhao is not None and resp.talhao.id == "t-9"
    assert message_repo.create.await_args_list[-1].kwargs["diagnosis_id"] == "diag-1"


async def test_resume_nao_repete_laudo_de_antes_da_pergunta(chat_service, session_repo, diagnosis_svc):
    session_repo.get_by_id.return_value = _fake_session("sess-1")
    graph = AsyncMock()
    graph.aget_state = AsyncMock(return_value=_snapshot({"diagnoses_in_turn": ["diag-1"], "talhao_selected": SEL}))
    graph.ainvoke = AsyncMock(
        return_value={
            "messages": [AIMessage(content="Anotado.")],
            "diagnoses_in_turn": ["diag-1"],
            "talhao_selected": SEL,
        }
    )
    diagnosis_svc.get_by_id = AsyncMock()
    with patch("app.domains.chat.service.build_graph", return_value=graph):
        resp = await chat_service.resume("u-1", "sess-1", "menos de 15 dias")
    assert resp.diagnosis is None and resp.talhao is None
    diagnosis_svc.get_by_id.assert_not_called()


async def test_resume_stream_emite_laudo_e_talhao_novos(chat_service, session_repo, message_repo, diagnosis_svc):
    session_repo.get_by_id.return_value = _fake_session("sess-1")
    diagnosis_svc.get_by_id = AsyncMock(return_value=_fake_diagnosis_response("diag-1"))
    graph = MagicMock()

    async def _events(*a, **k):
        return
        yield  # pragma: no cover

    graph.astream_events = _events
    graph.aget_state = AsyncMock(
        side_effect=[
            _snapshot({}),  # antes do resume
            _snapshot({"diagnoses_in_turn": ["diag-1"], "talhao_selected": SEL}),  # depois
        ]
    )
    with patch("app.domains.chat.service.build_graph", return_value=graph), patch.object(
        ChatService, "_final_text_from_snapshot", AsyncMock(return_value="ok")
    ), patch.object(ChatService, "_extract_pending_interrupt", AsyncMock(return_value=None)):
        events = await _drain(chat_service.resume_stream("u-1", "sess-1", "Talhão 9 · Rio"))
    kinds = [e["event"] for e in events]
    assert "talhao" in kinds and "diagnosis" in kinds and kinds[-1] == "done"
    assert message_repo.create.await_args_list[-1].kwargs["diagnosis_id"] == "diag-1"
