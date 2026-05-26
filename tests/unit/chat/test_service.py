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


# ── TCC-051: plan_features resolution ────────────────────────────────────────


async def test_resolve_plan_features_falls_back_to_free_when_no_sub_repo(chat_service):
    """Sem sub_repo, sempre FREE_FEATURES."""
    from app.domains.subscriptions.features import FREE_FEATURES

    result = await chat_service._resolve_plan_features("user-1")
    assert result == FREE_FEATURES


async def test_resolve_plan_features_falls_back_to_free_when_no_subscription(
    session_repo, message_repo, inference_svc, action_plan_svc, diagnosis_svc
):
    from app.domains.subscriptions.features import FREE_FEATURES

    sub_repo = AsyncMock()
    sub_repo.find_user_subscription = AsyncMock(return_value=None)

    svc = ChatService(
        session_repo=session_repo,
        message_repo=message_repo,
        inference_svc=inference_svc,
        action_plan_svc=action_plan_svc,
        diagnosis_svc=diagnosis_svc,
        sub_repo=sub_repo,
    )
    result = await svc._resolve_plan_features("user-1")
    assert result == FREE_FEATURES


async def test_resolve_plan_features_parses_pro(
    session_repo, message_repo, inference_svc, action_plan_svc, diagnosis_svc
):
    from app.domains.subscriptions.dto import PlanDTO, SubscriptionDTO
    from app.domains.subscriptions.features import PRO_FEATURES

    plan_dto = PlanDTO(
        id="p-1",
        name="pro",
        display_name="Pro",
        chat_daily_limit=None,
        inference_daily_limit=None,
        api_monthly_limit=500,
        is_active=True,
        features=PRO_FEATURES.model_dump(),
    )
    sub_dto = SubscriptionDTO(
        id="s-1",
        user_id="user-1",
        plan=plan_dto,
        started_at=NOW,
        expires_at=None,
        is_active=True,
    )
    sub_repo = AsyncMock()
    sub_repo.find_user_subscription = AsyncMock(return_value=sub_dto)

    svc = ChatService(
        session_repo=session_repo,
        message_repo=message_repo,
        inference_svc=inference_svc,
        action_plan_svc=action_plan_svc,
        diagnosis_svc=diagnosis_svc,
        sub_repo=sub_repo,
    )
    result = await svc._resolve_plan_features("user-1")
    assert result.tier_name == "pro"
    assert result.llm_model == "openai:gpt-4o"


async def test_resolve_plan_features_resilient_to_bad_features_dict(
    session_repo, message_repo, inference_svc, action_plan_svc, diagnosis_svc
):
    """Quando features tem keys invalidos, cai em FREE_FEATURES (resilient)."""
    from app.domains.subscriptions.dto import PlanDTO, SubscriptionDTO
    from app.domains.subscriptions.features import FREE_FEATURES

    plan_dto = PlanDTO(
        id="p-1",
        name="weird",
        display_name="Weird",
        chat_daily_limit=None,
        inference_daily_limit=None,
        api_monthly_limit=None,
        is_active=True,
        features={"invalid_key": True},  # falta tier_name -> ValidationError
    )
    sub_dto = SubscriptionDTO(
        id="s-1",
        user_id="user-1",
        plan=plan_dto,
        started_at=NOW,
        expires_at=None,
        is_active=True,
    )
    sub_repo = AsyncMock()
    sub_repo.find_user_subscription = AsyncMock(return_value=sub_dto)

    svc = ChatService(
        session_repo=session_repo,
        message_repo=message_repo,
        inference_svc=inference_svc,
        action_plan_svc=action_plan_svc,
        diagnosis_svc=diagnosis_svc,
        sub_repo=sub_repo,
    )
    result = await svc._resolve_plan_features("user-1")
    assert result == FREE_FEATURES


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

    async def _capture(state, config=None):
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

    async def _capture(state, config=None):
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


# ── Sprint A4.5: resume + interrupts (TCC-058) ───────────────────────────────


def _interrupt_obj(value: dict):
    """Cria um stub que simula ``langgraph.types.Interrupt`` com ``.value``."""
    obj = MagicMock()
    obj.value = value
    return obj


def _snapshot_with_interrupts(payload: dict | None, created_at=None):
    """Cria mock de StateSnapshot com tasks/interrupts populados."""
    snap = MagicMock()
    if payload is None:
        snap.tasks = []
    else:
        task = MagicMock()
        task.interrupts = [_interrupt_obj(payload)]
        snap.tasks = [task]
    snap.created_at = created_at
    return snap


async def test_extract_interrupt_from_result_returns_info():
    """Resultado com __interrupt__ deve virar InterruptInfo."""
    raw = {
        "__interrupt__": (
            _interrupt_obj(
                {
                    "kind": "ask_user",
                    "question": "Qual cultivo?",
                    "response_kind": "choice",
                    "options": ["soja", "milho"],
                    "asked_at": "2026-05-22T10:00:00+00:00",
                }
            ),
        )
    }
    info = ChatService._extract_interrupt_from_result(raw)
    assert info is not None
    assert info.question == "Qual cultivo?"
    assert info.response_kind == "choice"
    assert info.options == ["soja", "milho"]


async def test_extract_interrupt_from_result_returns_none_when_absent():
    assert ChatService._extract_interrupt_from_result({"messages": []}) is None
    assert ChatService._extract_interrupt_from_result(None) is None
    assert (
        ChatService._extract_interrupt_from_result(
            {"__interrupt__": ()}
        )
        is None
    )


async def test_chat_returns_interrupt_when_graph_pauses(
    chat_service, message_repo
):
    """chat() retorna ChatResponse com interrupt=info quando grafo pausa."""
    graph = AsyncMock()
    graph.ainvoke = AsyncMock(
        return_value={
            "messages": [],
            "__interrupt__": (
                _interrupt_obj(
                    {
                        "kind": "ask_user",
                        "question": "Confirma?",
                        "response_kind": "boolean",
                    }
                ),
            ),
        }
    )

    with patch("app.domains.chat.service.build_graph", return_value=graph):
        resp = await chat_service.chat(
            user_id="user-1",
            session_id=None,
            message_text="hi",
            image_filename=None,
            model_id="ensemble",
        )

    assert resp.interrupt is not None
    assert resp.interrupt.question == "Confirma?"
    assert resp.content == ""
    # Apenas user msg foi persistida — assistant nao (esta esperando resume)
    assert message_repo.create.await_count == 1


async def test_resume_invokes_command_resume(chat_service, session_repo, message_repo):
    """resume() chama graph.ainvoke(Command(resume=resposta)) e persiste msgs."""
    from langgraph.types import Command

    session_repo.get_by_id.return_value = _fake_session("sess-1")
    captured: list = []

    async def _capture(payload, config=None):
        captured.append(payload)
        return {"messages": [AIMessage(content="ok apos resume")]}

    graph = AsyncMock()
    graph.ainvoke = _capture

    with patch("app.domains.chat.service.build_graph", return_value=graph):
        resp = await chat_service.resume("user-1", "sess-1", "soja")

    assert resp.content == "ok apos resume"
    assert resp.session_id == "sess-1"
    assert captured
    assert isinstance(captured[0], Command)
    # User resume msg + assistant final
    assert message_repo.create.await_count == 2
    user_call = message_repo.create.await_args_list[0]
    assert user_call.kwargs["content"] == "soja"
    assert user_call.kwargs["metadata"] == {"resume": True}


async def test_resume_returns_chained_interrupt(chat_service, session_repo):
    """Se o resume dispara outra pergunta, body devolve interrupt populado."""
    session_repo.get_by_id.return_value = _fake_session("sess-1")
    graph = AsyncMock()
    graph.ainvoke = AsyncMock(
        return_value={
            "messages": [],
            "__interrupt__": (
                _interrupt_obj(
                    {
                        "kind": "ask_user",
                        "question": "E o nivel?",
                        "response_kind": "choice",
                        "options": ["essencial", "campo"],
                    }
                ),
            ),
        }
    )

    with patch("app.domains.chat.service.build_graph", return_value=graph):
        resp = await chat_service.resume("user-1", "sess-1", "soja")

    assert resp.interrupt is not None
    assert resp.interrupt.options == ["essencial", "campo"]
    assert resp.content == ""


async def test_resume_missing_session_returns_empty(chat_service, session_repo):
    session_repo.get_by_id.return_value = None
    resp = await chat_service.resume("user-1", "missing", "x")
    assert resp.content == ""
    assert resp.session_id == "missing"


async def test_list_pending_interrupts_returns_empty_without_checkpointer(
    session_repo, message_repo, inference_svc, action_plan_svc, diagnosis_svc,
):
    svc = ChatService(
        session_repo=session_repo,
        message_repo=message_repo,
        inference_svc=inference_svc,
        action_plan_svc=action_plan_svc,
        diagnosis_svc=diagnosis_svc,
    )
    assert await svc.list_pending_interrupts("user-1") == []


async def test_list_pending_interrupts_filters_sessions_with_interrupt(
    session_repo, message_repo, inference_svc, action_plan_svc, diagnosis_svc,
):
    """Sessoes sem interrupt sao ignoradas; com interrupt entram na lista."""
    ckpt_factory = AsyncMock(return_value=MagicMock())
    svc = ChatService(
        session_repo=session_repo,
        message_repo=message_repo,
        inference_svc=inference_svc,
        action_plan_svc=action_plan_svc,
        diagnosis_svc=diagnosis_svc,
        checkpointer_factory=ckpt_factory,
    )

    session_repo.list_for_user = AsyncMock(
        return_value=[
            _fake_session("sess-A"),
            _fake_session("sess-B"),
            _fake_session("sess-C"),
        ]
    )

    snap_with = _snapshot_with_interrupts(
        {
            "kind": "ask_user",
            "question": "Qual cultivo?",
            "response_kind": "choice",
            "options": ["soja"],
            "asked_at": "2026-05-22T12:00:00+00:00",
        },
        created_at="2026-05-22T12:00:00+00:00",
    )
    snap_without = _snapshot_with_interrupts(None)

    async def _aget_state(config):
        thread_id = config["configurable"]["thread_id"]
        if thread_id == "sess-A":
            return snap_with
        if thread_id == "sess-B":
            return snap_without
        # sess-C — simula erro de leitura: deve ser pulada
        raise RuntimeError("snapshot missing")

    graph = MagicMock()
    graph.aget_state = _aget_state

    with patch("app.domains.chat.service.build_graph", return_value=graph):
        result = await svc.list_pending_interrupts("user-1")

    assert len(result) == 1
    assert result[0].session_id == "sess-A"
    assert result[0].interrupt.question == "Qual cultivo?"
    assert result[0].interrupt.options == ["soja"]


async def test_list_pending_interrupts_skips_invalid_payload(
    session_repo, message_repo, inference_svc, action_plan_svc, diagnosis_svc,
):
    """Interrupt cujo value nao eh dict eh silenciosamente pulado."""
    ckpt_factory = AsyncMock(return_value=MagicMock())
    svc = ChatService(
        session_repo=session_repo,
        message_repo=message_repo,
        inference_svc=inference_svc,
        action_plan_svc=action_plan_svc,
        diagnosis_svc=diagnosis_svc,
        checkpointer_factory=ckpt_factory,
    )
    session_repo.list_for_user = AsyncMock(
        return_value=[_fake_session("sess-X")]
    )

    snap = _snapshot_with_interrupts({"question": "ok"})  # falta required fields
    # Substitui value por string pra forcar branch invalid
    snap.tasks[0].interrupts[0].value = "not a dict"

    graph = MagicMock()
    graph.aget_state = AsyncMock(return_value=snap)

    with patch("app.domains.chat.service.build_graph", return_value=graph):
        result = await svc.list_pending_interrupts("user-1")

    assert result == []


async def test_get_graph_caches_compiled_instance(chat_service):
    """_get_graph deve memoizar o grafo entre chamadas."""
    graph = MagicMock()
    with patch(
        "app.domains.chat.service.build_graph", return_value=graph
    ) as bg:
        g1 = await chat_service._get_graph()
        g2 = await chat_service._get_graph()
        assert g1 is g2 is graph
        assert bg.call_count == 1


async def test_get_graph_handles_checkpointer_factory_failure(
    session_repo, message_repo, inference_svc, action_plan_svc, diagnosis_svc,
):
    """Falha do checkpointer_factory nao bloqueia build_graph."""
    failing = AsyncMock(side_effect=RuntimeError("no postgres"))
    svc = ChatService(
        session_repo=session_repo,
        message_repo=message_repo,
        inference_svc=inference_svc,
        action_plan_svc=action_plan_svc,
        diagnosis_svc=diagnosis_svc,
        checkpointer_factory=failing,
    )
    graph = MagicMock()
    with patch("app.domains.chat.service.build_graph", return_value=graph) as bg:
        g = await svc._get_graph()
        assert g is graph
        # build_graph foi chamado com checkpointer=None
        assert bg.call_args.kwargs["checkpointer"] is None
