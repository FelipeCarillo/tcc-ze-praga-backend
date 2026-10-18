"""TCC-100 / TCC-102 — achados do teste de UX de 17/10/2026.

- Streaming com a Responses API (gpt-6): o chunk traz lista de blocos.
- Laudo gerado antes de uma pergunta do Zé vai junto da pergunta.
- Mensagem nova com pergunta pendente vira a resposta dela.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk
from langgraph.types import Command

from app.domains.chat.service import ChatService
from tests.unit.chat.test_service import (
    _drain,
    _fake_diagnosis_response,
    _fake_session,
    _interrupt_obj,
    _snapshot_with_interrupts,
)

PERGUNTA = {
    "kind": "ask_user",
    "question": "Antes de analisar: em qual talhão você tirou essa foto?",
    "response_kind": "choice",
    "options": ["Talhão 3", "Novo talhão", "Pular"],
}


@pytest.fixture
def session_repo():
    repo = AsyncMock()
    repo.get_or_create_for_user.return_value = _fake_session()
    repo.get_by_id.return_value = _fake_session()
    return repo


@pytest.fixture
def message_repo():
    repo = AsyncMock()
    repo.create.return_value = MagicMock()
    return repo


@pytest.fixture
def diagnosis_svc():
    svc = AsyncMock()
    svc.get_by_id = AsyncMock(return_value=_fake_diagnosis_response("diag-1"))
    return svc


@pytest.fixture
def chat_service(session_repo, message_repo, diagnosis_svc):
    return ChatService(
        session_repo=session_repo,
        message_repo=message_repo,
        inference_svc=MagicMock(),
        action_plan_svc=AsyncMock(),
        diagnosis_svc=diagnosis_svc,
    )


def _snap(values: dict, payload: dict | None = None):
    snap = _snapshot_with_interrupts(payload)
    snap.values = values
    return snap


# ── TCC-100: streaming com blocos ────────────────────────────────────────────


def test_chunk_text_ignora_raciocinio_e_tool_call():
    chunk = AIMessageChunk(
        content=[
            {"type": "reasoning", "encrypted_content": "gAAAA..."},
            {"type": "function_call", "name": "inspect_image", "arguments": "{}"},
            {"type": "text", "text": "Olá, ", "index": 0},
            {"type": "text", "text": "produtor!", "index": 0},
        ]
    )
    assert ChatService._chunk_text(chunk) == "Olá, produtor!"
    assert ChatService._chunk_text(AIMessageChunk(content="texto puro")) == "texto puro"
    assert ChatService._chunk_text(None) == ""


async def test_stream_com_blocos_emite_so_texto_e_salva(chat_service, message_repo):
    graph = MagicMock()

    async def _events(*a, **k):
        for content in (
            [{"type": "reasoning", "encrypted_content": "x"}],
            [{"type": "text", "text": "Ferrugem ", "index": 0}],
            [{"type": "text", "text": "provável.", "index": 0}],
        ):
            yield {
                "event": "on_chat_model_stream",
                "metadata": {"langgraph_node": "llm"},
                "data": {"chunk": AIMessageChunk(content=content)},
            }

    graph.astream_events = _events
    graph.aget_state = AsyncMock(return_value=_snap({}))
    with patch("app.domains.chat.service.build_graph", return_value=graph):
        events = await _drain(
            chat_service.chat_stream(
                user_id="u-1", session_id=None, message_text="oi", image_bytes=None,
                image_mime=None, image_filename=None, model_id="ensemble",
            )
        )
    tokens = [e["data"] for e in events if e["event"] == "token"]
    assert tokens == ["Ferrugem ", "provável."]
    assert all(isinstance(t, str) for t in tokens)
    assert "error" not in [e["event"] for e in events]
    assert message_repo.create.await_args_list[-1].kwargs["content"] == "Ferrugem provável."


# ── TCC-102: laudo junto da pergunta ─────────────────────────────────────────


async def test_chat_com_pergunta_depois_do_laudo_devolve_os_dois(chat_service, message_repo):
    graph = AsyncMock()
    graph.ainvoke = AsyncMock(
        return_value={
            "messages": [],
            "diagnoses_in_turn": ["diag-1"],
            "__interrupt__": (_interrupt_obj(PERGUNTA),),
        }
    )
    with patch("app.domains.chat.service.build_graph", return_value=graph):
        resp = await chat_service.chat("u-1", None, "olha a folha", None, None, None, model_id="ensemble")
    assert resp.interrupt is not None and resp.diagnosis is not None
    assert resp.diagnosis.id == "diag-1"
    # a pergunta é gravada com o laudo, para o cartão voltar no histórico
    saved = message_repo.create.await_args_list[-1].kwargs
    assert saved["role"] == "assistant" and saved["diagnosis_id"] == "diag-1"
    assert saved["content"] == PERGUNTA["question"]


async def test_resume_com_segunda_pergunta_traz_laudo_novo(chat_service, diagnosis_svc):
    graph = AsyncMock()
    graph.aget_state = AsyncMock(return_value=_snap({"diagnoses_in_turn": []}))
    graph.ainvoke = AsyncMock(
        return_value={
            "messages": [],
            "diagnoses_in_turn": ["diag-1"],
            "__interrupt__": (_interrupt_obj({**PERGUNTA, "question": "Tem pústulas no verso?"}),),
        }
    )
    with patch("app.domains.chat.service.build_graph", return_value=graph):
        resp = await chat_service.resume("u-1", "sess-1", "Pular")
    assert resp.interrupt is not None and resp.diagnosis is not None


async def test_pergunta_sem_laudo_nao_grava_mensagem(chat_service, message_repo):
    graph = AsyncMock()
    graph.ainvoke = AsyncMock(
        return_value={"messages": [], "diagnoses_in_turn": [], "__interrupt__": (_interrupt_obj(PERGUNTA),)}
    )
    with patch("app.domains.chat.service.build_graph", return_value=graph):
        resp = await chat_service.chat("u-1", None, "olha", None, None, None, model_id="ensemble")
    assert resp.diagnosis is None
    assert [c.kwargs["role"] for c in message_repo.create.await_args_list] == ["user"]


async def test_stream_emite_laudo_antes_da_pergunta(chat_service):
    graph = MagicMock()

    async def _events(*a, **k):
        return
        yield  # pragma: no cover

    graph.astream_events = _events
    graph.aget_state = AsyncMock(return_value=_snap({"diagnoses_in_turn": ["diag-1"]}, PERGUNTA))
    with patch("app.domains.chat.service.build_graph", return_value=graph):
        events = await _drain(
            chat_service.chat_stream(
                user_id="u-1", session_id=None, message_text="olha", image_bytes=None,
                image_mime=None, image_filename=None, model_id="ensemble",
            )
        )
    kinds = [e["event"] for e in events]
    assert kinds.index("diagnosis") < kinds.index("interrupt")
    assert json.loads(next(e["data"] for e in events if e["event"] == "diagnosis"))["id"] == "diag-1"


# ── TCC-102: mensagem nova com pergunta pendente ─────────────────────────────


async def test_mensagem_nova_com_pergunta_pendente_vira_resposta(chat_service):
    graph = AsyncMock()
    graph.aget_state = AsyncMock(return_value=_snap({}, PERGUNTA))
    graph.ainvoke = AsyncMock(return_value={"messages": [AIMessage(content="Ok, sem talhão.")]})
    with patch("app.domains.chat.service.build_graph", return_value=graph):
        resp = await chat_service.chat(
            "u-1", "sess-1", "deixa isso, é grave?", b"\xff\xd8img", "image/jpeg", "f.jpg",
            model_id="ensemble",
        )
    sent = graph.ainvoke.await_args.args[0]
    assert isinstance(sent, Command)
    assert sent.resume.startswith("deixa isso, é grave?") and "foto" in sent.resume
    assert set(sent.update) == {"uploaded_files"}  # foto nova entra; o resto do turno fica
    assert sent.update["uploaded_files"][0].mime == "image/jpeg"
    assert resp.content == "Ok, sem talhão."


async def test_mensagem_nova_sem_foto_preserva_a_foto_do_turno_pausado(chat_service):
    """Sem foto nova, a do turno pausado continua valendo (achado do re-teste)."""
    graph = AsyncMock()
    graph.aget_state = AsyncMock(return_value=_snap({}, PERGUNTA))
    graph.ainvoke = AsyncMock(return_value={"messages": [AIMessage(content="ok")]})
    with patch("app.domains.chat.service.build_graph", return_value=graph):
        await chat_service.chat(
            "u-1", "sess-1", "deixa o talhão pra lá", None, None, None,
            model_id="ensemble", talhao_id="t-9",
        )
    sent = graph.ainvoke.await_args.args[0]
    assert sent.resume == "deixa o talhão pra lá"
    assert sent.update == {"selected_talhao_id": "t-9"}


async def test_mensagem_nova_sem_pergunta_pendente_e_turno_normal(chat_service):
    graph = AsyncMock()
    graph.aget_state = AsyncMock(return_value=_snap({}))
    graph.ainvoke = AsyncMock(return_value={"messages": [AIMessage(content="oi")]})
    with patch("app.domains.chat.service.build_graph", return_value=graph):
        await chat_service.chat("u-1", "sess-1", "oi", None, None, None, model_id="ensemble")
    sent = graph.ainvoke.await_args.args[0]
    assert isinstance(sent, dict) and sent["messages"][0].content.startswith("oi")


async def test_sessao_nova_nem_consulta_pergunta_pendente(chat_service):
    graph = AsyncMock()
    graph.aget_state = AsyncMock()
    graph.ainvoke = AsyncMock(return_value={"messages": [AIMessage(content="oi")]})
    with patch("app.domains.chat.service.build_graph", return_value=graph):
        await chat_service.chat("u-1", None, "oi", None, None, None, model_id="ensemble")
    graph.aget_state.assert_not_awaited()


async def test_stream_mensagem_nova_com_pergunta_pendente_vira_resposta(chat_service):
    graph = MagicMock()
    seen = {}

    async def _events(inp, **k):
        seen["input"] = inp
        return
        yield  # pragma: no cover

    graph.astream_events = _events
    graph.aget_state = AsyncMock(side_effect=[_snap({}, PERGUNTA), _snap({}), _snap({}), _snap({}), _snap({})])
    with patch("app.domains.chat.service.build_graph", return_value=graph), patch.object(
        ChatService, "_final_text_from_snapshot", AsyncMock(return_value="ok")
    ):
        await _drain(
            chat_service.chat_stream(
                user_id="u-1", session_id="sess-1", message_text="Pular", image_bytes=None,
                image_mime=None, image_filename=None, model_id="ensemble",
            )
        )
    assert isinstance(seen["input"], Command) and seen["input"].resume == "Pular"


async def test_resume_stream_tokens_em_blocos_e_laudo_com_talhao_antes_da_pergunta(chat_service, message_repo):
    sel = {"id": "t-9", "nome": "Talhão 9", "apelido": None, "hectares": 40.0,
           "data_semeadura": None, "fazenda_id": "f-1", "fazenda_nome": "Boa Vista", "created": False}
    graph = MagicMock()

    async def _events(*a, **k):
        yield {
            "event": "on_chat_model_stream",
            "metadata": {"langgraph_node": "llm"},
            "data": {"chunk": AIMessageChunk(content=[{"type": "reasoning"}, {"type": "text", "text": "Vi a folha."}])},
        }
        yield {
            "event": "on_chat_model_stream",
            "metadata": {"langgraph_node": "tools"},
            "data": {"chunk": AIMessageChunk(content="vazaria")},
        }

    graph.astream_events = _events
    graph.aget_state = AsyncMock(
        side_effect=[
            _snap({}),  # antes do resume
            _snap({"diagnoses_in_turn": ["diag-1"], "talhao_selected": sel}, PERGUNTA),  # pendente?
            _snap({"diagnoses_in_turn": ["diag-1"], "talhao_selected": sel}, PERGUNTA),  # valores
        ]
    )
    with patch("app.domains.chat.service.build_graph", return_value=graph):
        events = await _drain(chat_service.resume_stream("u-1", "sess-1", "Talhão 9"))
    kinds = [e["event"] for e in events]
    assert [e["data"] for e in events if e["event"] == "token"] == ["Vi a folha."]
    assert kinds.index("talhao") < kinds.index("interrupt")
    assert kinds.index("diagnosis") < kinds.index("interrupt")
    assert message_repo.create.await_args_list[-1].kwargs["diagnosis_id"] == "diag-1"
