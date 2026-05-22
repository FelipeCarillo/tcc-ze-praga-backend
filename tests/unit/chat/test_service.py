"""Testes unitários do ChatService.

Mocka build_graph e todos os services downstream — focado na orquestração.
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.domains.chat.dto import ChatSessionDTO
from app.domains.chat.schemas import ChatResponse
from app.domains.chat.service import ChatService
from app.domains.diagnoses.schemas import DiagnosisResponse, Top3PredictionSchema
from app.domains.inference.schemas import InferenceResult
from app.shared.enums import SeverityEnum

NOW = datetime(2026, 5, 22, 12, 0, 0, tzinfo=UTC)


def _fake_session(id_: str = "sess-1") -> ChatSessionDTO:
    return ChatSessionDTO(
        id=id_,
        user_id="user-1",
        title=None,
        created_at=NOW,
        updated_at=NOW,
    )


def _fake_inference_result() -> InferenceResult:
    return InferenceResult(
        disease_id="ferrugem-asiatica",
        disease_name="Ferrugem Asiática",
        scientific_name="Phakopsora pachyrhizi",
        severity=SeverityEnum.ALTA,
        description="...",
        confidence=0.92,
        model_id="ensemble",
        image_name="f.jpg",
        top3=[
            Top3PredictionSchema(
                rank=1,
                disease_name="Ferrugem Asiática",
                disease_id="ferrugem-asiatica",
                scientific_name="Phakopsora pachyrhizi",
                confidence=0.92,
                severity="alta",
            )
        ],
    )


def _fake_diagnosis_response(id_: str = "diag-1") -> DiagnosisResponse:
    return DiagnosisResponse(
        id=id_,
        disease_name="Ferrugem Asiática",
        disease_id="ferrugem-asiatica",
        scientific_name="Phakopsora pachyrhizi",
        confidence=0.92,
        severity="alta",
        description="...",
        model_used="ensemble",
        image_url=None,
        image_name="f.jpg",
        created_at=NOW,
        top3=[
            Top3PredictionSchema(
                rank=1,
                disease_name="Ferrugem Asiática",
                disease_id="ferrugem-asiatica",
                scientific_name="Phakopsora pachyrhizi",
                confidence=0.92,
                severity="alta",
            )
        ],
    )


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
def inference_svc():
    svc = MagicMock()
    svc.predict.return_value = _fake_inference_result()
    return svc


@pytest.fixture
def action_plan_svc():
    return AsyncMock()


@pytest.fixture
def diagnosis_svc():
    svc = AsyncMock()
    svc.create.return_value = _fake_diagnosis_response()
    return svc


@pytest.fixture
def chat_service(session_repo, message_repo, inference_svc, action_plan_svc, diagnosis_svc):
    return ChatService(
        session_repo=session_repo,
        message_repo=message_repo,
        inference_svc=inference_svc,
        action_plan_svc=action_plan_svc,
        diagnosis_svc=diagnosis_svc,
    )


def _graph_returning(content: str):
    """Helper: cria mock de graph cujo ainvoke retorna estado com AIMessage(content)."""
    graph = AsyncMock()
    graph.ainvoke = AsyncMock(
        return_value={"messages": [HumanMessage(content="u"), AIMessage(content=content)]}
    )
    return graph


# ── chat() ────────────────────────────────────────────────────────────────────


async def test_chat_text_only_persists_user_and_assistant_messages(
    chat_service, session_repo, message_repo, inference_svc, diagnosis_svc
):
    with patch(
        "app.domains.chat.service.build_graph",
        return_value=_graph_returning("oi de volta"),
    ):
        resp = await chat_service.chat(
            user_id="user-1",
            session_id=None,
            message_text="oi",
            image_filename=None,
            model_id="ensemble",
        )

    assert isinstance(resp, ChatResponse)
    assert resp.content == "oi de volta"
    assert resp.diagnosis is None
    assert resp.session_id == "sess-1"

    # Persistiu user + assistant
    assert message_repo.create.await_count == 2
    inference_svc.predict.assert_not_called()
    diagnosis_svc.create.assert_not_called()


async def test_chat_with_image_runs_inference_and_persists_diagnosis(
    chat_service, session_repo, message_repo, inference_svc, diagnosis_svc
):
    with patch(
        "app.domains.chat.service.build_graph",
        return_value=_graph_returning("Detectei Ferrugem."),
    ):
        resp = await chat_service.chat(
            user_id="user-1",
            session_id=None,
            message_text="analisa",
            image_filename="folha.jpg",
            model_id="ensemble",
        )

    assert resp.diagnosis is not None
    assert resp.diagnosis.disease_id == "ferrugem-asiatica"

    inference_svc.predict.assert_called_once_with("ensemble", "folha.jpg")
    diagnosis_svc.create.assert_awaited_once()

    # 2 mensagens: user + assistant. A assistant deve linkar o diagnosis.
    assert message_repo.create.await_count == 2
    assistant_call = message_repo.create.await_args_list[1]
    assert assistant_call.kwargs["diagnosis_id"] == "diag-1"


async def test_chat_uses_existing_session_id(chat_service, session_repo, message_repo):
    session_repo.get_or_create_for_user.return_value = _fake_session("existing")
    with patch(
        "app.domains.chat.service.build_graph",
        return_value=_graph_returning("..."),
    ):
        resp = await chat_service.chat(
            user_id="user-1",
            session_id="existing",
            message_text="oi",
            image_filename=None,
            model_id="vit",
        )
    assert resp.session_id == "existing"
    session_repo.get_or_create_for_user.assert_awaited_once_with("user-1", "existing")


async def test_chat_user_message_metadata_includes_image_filename(chat_service, message_repo):
    with patch(
        "app.domains.chat.service.build_graph",
        return_value=_graph_returning("ok"),
    ):
        await chat_service.chat(
            user_id="user-1",
            session_id=None,
            message_text="analisa",
            image_filename="x.jpg",
            model_id="ensemble",
        )
    user_msg_call = message_repo.create.await_args_list[0]
    assert user_msg_call.kwargs["metadata"] == {"image_filename": "x.jpg"}


async def test_chat_user_message_metadata_none_when_no_image(chat_service, message_repo):
    with patch(
        "app.domains.chat.service.build_graph",
        return_value=_graph_returning("ok"),
    ):
        await chat_service.chat(
            user_id="user-1",
            session_id=None,
            message_text="oi",
            image_filename=None,
            model_id="ensemble",
        )
    user_msg_call = message_repo.create.await_args_list[0]
    assert user_msg_call.kwargs["metadata"] is None


# ── _extract_final_text ───────────────────────────────────────────────────────


def test_extract_final_text_from_simple_string():
    out = ChatService._extract_final_text(
        [HumanMessage(content="u"), AIMessage(content="resposta final")]
    )
    assert out == "resposta final"


def test_extract_final_text_skips_intermediate_ai():
    out = ChatService._extract_final_text(
        [
            HumanMessage(content="u"),
            AIMessage(content=""),  # tool call sem texto
            AIMessage(content="final"),
        ]
    )
    assert out == "final"


def test_extract_final_text_empty_when_no_ai():
    assert ChatService._extract_final_text([HumanMessage(content="u")]) == ""


def test_extract_final_text_handles_list_content():
    msg = AIMessage(content=[{"text": "parte 1"}, {"text": " parte 2"}])
    out = ChatService._extract_final_text([msg])
    assert out == "parte 1 parte 2"


# ── _tool_output_to_text ──────────────────────────────────────────────────────


def test_tool_output_text_none():
    assert ChatService._tool_output_to_text(None) == ""


def test_tool_output_text_string():
    assert ChatService._tool_output_to_text("hello") == "hello"


def test_tool_output_text_object_with_content_string():
    obj = MagicMock(spec=["content"])
    obj.content = "abc"
    assert ChatService._tool_output_to_text(obj) == "abc"


def test_tool_output_text_object_with_content_dict():
    obj = MagicMock(spec=["content"])
    obj.content = {"key": "value"}
    out = ChatService._tool_output_to_text(obj)
    assert "key" in out and "value" in out


def test_tool_output_text_fallback_str():
    assert ChatService._tool_output_to_text(42) == "42"


# ── chat_stream() ─────────────────────────────────────────────────────────────


async def _drain(agen):
    return [e async for e in agen]


async def test_chat_stream_emits_done_event(chat_service, message_repo):
    graph = MagicMock()

    async def _empty_events(*args, **kwargs):
        return
        yield  # pragma: no cover

    graph.astream_events = _empty_events
    graph.ainvoke = AsyncMock(
        return_value={"messages": [AIMessage(content="resposta consolidada")]}
    )

    with patch("app.domains.chat.service.build_graph", return_value=graph):
        events = await _drain(
            chat_service.chat_stream(
                user_id="user-1",
                session_id=None,
                message_text="oi",
                image_filename=None,
                model_id="ensemble",
            )
        )

    assert events[-1]["event"] == "done"
    # User + assistant mensagens persistidas
    assert message_repo.create.await_count == 2


async def test_chat_stream_emits_token_events(chat_service):
    graph = MagicMock()

    async def _events_with_tokens(*args, **kwargs):
        chunk1 = MagicMock()
        chunk1.content = "Hello "
        chunk2 = MagicMock()
        chunk2.content = "world"
        yield {"event": "on_chat_model_stream", "data": {"chunk": chunk1}}
        yield {"event": "on_chat_model_stream", "data": {"chunk": chunk2}}

    graph.astream_events = _events_with_tokens

    with patch("app.domains.chat.service.build_graph", return_value=graph):
        events = await _drain(
            chat_service.chat_stream(
                user_id="user-1",
                session_id=None,
                message_text="oi",
                image_filename=None,
                model_id="ensemble",
            )
        )

    tokens = [e["data"] for e in events if e["event"] == "token"]
    assert tokens == ["Hello ", "world"]


async def test_chat_stream_emits_tool_events(chat_service):
    graph = MagicMock()

    async def _events_with_tools(*args, **kwargs):
        yield {"event": "on_tool_start", "name": "analyze_image", "data": {}}
        tool_out = MagicMock(spec=["content"])
        tool_out.content = "analysis result"
        yield {"event": "on_tool_end", "name": "analyze_image", "data": {"output": tool_out}}

    graph.astream_events = _events_with_tools
    # Sem tokens → cai no fallback ainvoke
    graph.ainvoke = AsyncMock(return_value={"messages": [AIMessage(content="resposta")]})

    with patch("app.domains.chat.service.build_graph", return_value=graph):
        events = await _drain(
            chat_service.chat_stream(
                user_id="user-1",
                session_id=None,
                message_text="x",
                image_filename=None,
                model_id="ensemble",
            )
        )

    kinds = [e["event"] for e in events]
    assert "tool_call" in kinds
    assert "tool_result" in kinds
    tool_call = next(e for e in events if e["event"] == "tool_call")
    assert tool_call["data"] == "analyze_image"


async def test_chat_stream_with_image_emits_diagnosis_event(
    chat_service, inference_svc, diagnosis_svc
):
    graph = MagicMock()

    async def _empty_events(*args, **kwargs):
        return
        yield  # pragma: no cover

    graph.astream_events = _empty_events
    graph.ainvoke = AsyncMock(return_value={"messages": [AIMessage(content="ok")]})

    with patch("app.domains.chat.service.build_graph", return_value=graph):
        events = await _drain(
            chat_service.chat_stream(
                user_id="user-1",
                session_id=None,
                message_text="analisa",
                image_filename="folha.jpg",
                model_id="ensemble",
            )
        )

    diag_events = [e for e in events if e["event"] == "diagnosis"]
    assert len(diag_events) == 1
    inference_svc.predict.assert_called_once_with("ensemble", "folha.jpg")
    diagnosis_svc.create.assert_awaited_once()


# ── TCC-048: prefetch + close_session ─────────────────────────────────────────


@pytest.fixture
def store_factory_fixture():
    store = AsyncMock()
    store.asearch = AsyncMock(return_value=[])
    factory = AsyncMock(return_value=store)
    return factory, store


async def test_chat_prefetches_relevant_diagnoses_when_store_present(
    session_repo, message_repo, inference_svc, action_plan_svc, diagnosis_svc,
    store_factory_fixture,
):
    """Quando store_factory eh injetado, chat() faz prefetch + injeta no state."""
    factory, store = store_factory_fixture
    fake_item = MagicMock()
    fake_item.value = {
        "summary_text": "Ferrugem em 2025-01",
        "diagnosis_id": "diag-old",
    }
    store.asearch.return_value = [fake_item]

    svc = ChatService(
        session_repo=session_repo,
        message_repo=message_repo,
        inference_svc=inference_svc,
        action_plan_svc=action_plan_svc,
        diagnosis_svc=diagnosis_svc,
        store_factory=factory,
    )

    captured_state: dict = {}
    graph = AsyncMock()

    async def _capture(state):
        captured_state.update(state)
        return {"messages": [AIMessage(content="ok")]}

    graph.ainvoke = _capture

    with patch("app.domains.chat.service.build_graph", return_value=graph):
        await svc.chat(
            user_id="user-1",
            session_id=None,
            message_text="ferrugem ano passado",
            image_filename=None,
            model_id="ensemble",
        )

    # Store.asearch foi chamado com namespace correto
    store.asearch.assert_awaited_once()
    call_args = store.asearch.call_args
    assert call_args.args[0] == ("user", "user-1", "diagnoses")
    assert call_args.kwargs["query"] == "ferrugem ano passado"

    # State recebeu o resultado
    assert "recent_relevant_diagnoses" in captured_state
    assert len(captured_state["recent_relevant_diagnoses"]) == 1
    assert (
        captured_state["recent_relevant_diagnoses"][0]["diagnosis_id"]
        == "diag-old"
    )


async def test_chat_prefetch_swallow_errors(
    session_repo, message_repo, inference_svc, action_plan_svc, diagnosis_svc,
):
    """Erro do Store no prefetch nao quebra o chat."""
    factory = AsyncMock()
    factory.return_value = MagicMock()
    factory.return_value.asearch = AsyncMock(
        side_effect=RuntimeError("offline")
    )

    svc = ChatService(
        session_repo=session_repo,
        message_repo=message_repo,
        inference_svc=inference_svc,
        action_plan_svc=action_plan_svc,
        diagnosis_svc=diagnosis_svc,
        store_factory=factory,
    )

    with patch(
        "app.domains.chat.service.build_graph",
        return_value=_graph_returning("ok"),
    ):
        resp = await svc.chat(
            user_id="user-1",
            session_id=None,
            message_text="oi",
            image_filename=None,
            model_id="ensemble",
        )

    assert resp.content == "ok"


async def test_chat_skips_prefetch_when_no_store(chat_service):
    """Sem store_factory, prefetch retorna [] e nao bate em nada."""
    captured_state: dict = {}
    graph = AsyncMock()

    async def _capture(state):
        captured_state.update(state)
        return {"messages": [AIMessage(content="ok")]}

    graph.ainvoke = _capture

    with patch("app.domains.chat.service.build_graph", return_value=graph):
        await chat_service.chat(
            user_id="user-1",
            session_id=None,
            message_text="oi",
            image_filename=None,
            model_id="ensemble",
        )

    assert captured_state.get("recent_relevant_diagnoses", []) == []


async def test_close_session_generates_and_persists_summary(
    session_repo, message_repo, inference_svc, action_plan_svc, diagnosis_svc,
    store_factory_fixture,
):
    factory, store = store_factory_fixture
    svc = ChatService(
        session_repo=session_repo,
        message_repo=message_repo,
        inference_svc=inference_svc,
        action_plan_svc=action_plan_svc,
        diagnosis_svc=diagnosis_svc,
        store_factory=factory,
    )

    session_repo.get_by_id.return_value = _fake_session("sess-1")
    fake_msgs = [
        MagicMock(role="user", content="ola"),
        MagicMock(role="assistant", content="ola de volta"),
    ]
    message_repo.list_by_session.return_value = fake_msgs
    session_repo.update_summary.return_value = _fake_session("sess-1")

    fake_llm = AsyncMock()
    fake_llm.ainvoke.return_value = AIMessage(content="Resumo: conversa breve.")

    result = await svc.close_session("user-1", "sess-1", llm=fake_llm)

    assert result.session_id == "sess-1"
    assert "Resumo" in result.summary_text
    session_repo.update_summary.assert_awaited_once()
    update_kwargs = session_repo.update_summary.await_args
    assert update_kwargs.args[0] == "sess-1"
    assert update_kwargs.args[1] == "user-1"
    # Store indexou o summary
    store.aput.assert_awaited_once()
    aput_kwargs = store.aput.call_args.kwargs
    assert aput_kwargs["namespace"] == ("user", "user-1", "session_summaries")
    assert aput_kwargs["key"] == "sess-1"


async def test_close_session_returns_empty_when_session_missing(
    session_repo, message_repo, inference_svc, action_plan_svc, diagnosis_svc,
):
    svc = ChatService(
        session_repo=session_repo,
        message_repo=message_repo,
        inference_svc=inference_svc,
        action_plan_svc=action_plan_svc,
        diagnosis_svc=diagnosis_svc,
    )
    session_repo.get_by_id.return_value = None

    result = await svc.close_session("user-1", "missing-sess")
    assert result.summary_text is None


async def test_close_session_handles_empty_messages(
    session_repo, message_repo, inference_svc, action_plan_svc, diagnosis_svc,
):
    svc = ChatService(
        session_repo=session_repo,
        message_repo=message_repo,
        inference_svc=inference_svc,
        action_plan_svc=action_plan_svc,
        diagnosis_svc=diagnosis_svc,
    )
    session_repo.get_by_id.return_value = _fake_session("sess-empty")
    message_repo.list_by_session.return_value = []

    result = await svc.close_session("user-1", "sess-empty")
    assert result.summary_text is None
    session_repo.update_summary.assert_not_awaited()
