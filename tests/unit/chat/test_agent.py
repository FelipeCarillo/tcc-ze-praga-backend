"""Testes do grafo LangGraph do Zé Praga (TCC-009).

Estratégia:
    - FakeToolLLM subclasseia FakeMessagesListChatModel sobrescrevendo
      bind_tools (a base levanta NotImplementedError). Sequência de respostas:
      primeiro turno chama tool, segundo turno consolida a resposta final.
    - Services mockados (InferenceService, ActionPlanService) — não dependemos
      do banco nem do OpenAI real.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage

from app.domains.action_plans.schemas import (
    ActionPlanLevelResponse,
    ActionPlanResponse,
    SourceResponse,
)
from app.domains.chat.agent import (
    SYSTEM_PROMPT,
    ChatState,
    build_graph,
)
from app.domains.diagnoses.schemas import Top3PredictionSchema
from app.domains.inference.schemas import InferenceResult
from app.shared.enums import ActionPlanLevelEnum, SeverityEnum


class FakeToolLLM(FakeMessagesListChatModel):
    """FakeMessagesListChatModel + bind_tools no-op (a base não implementa)."""

    def bind_tools(self, tools, **kwargs):  # type: ignore[override]
        return self


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_inference_result(
    disease_id: str = "ferrugem-asiatica",
    disease_name: str = "Ferrugem Asiática",
) -> InferenceResult:
    return InferenceResult(
        disease_id=disease_id,
        disease_name=disease_name,
        scientific_name="Phakopsora pachyrhizi",
        severity=SeverityEnum.ALTA,
        description="Doença severa.",
        confidence=0.93,
        model_id="ensemble",
        image_name="folha.jpg",
        top3=[
            Top3PredictionSchema(
                rank=1,
                disease_name=disease_name,
                disease_id=disease_id,
                scientific_name="Phakopsora pachyrhizi",
                confidence=0.93,
                severity="alta",
            ),
            Top3PredictionSchema(
                rank=2,
                disease_name="Mancha-Alvo",
                disease_id="mancha-alvo",
                scientific_name="Corynespora cassiicola",
                confidence=0.05,
                severity="media",
            ),
            Top3PredictionSchema(
                rank=3,
                disease_name="Antracnose",
                disease_id="antracnose",
                scientific_name="Colletotrichum truncatum",
                confidence=0.02,
                severity="media",
            ),
        ],
    )


def _make_action_plan_response(disease_id: str = "ferrugem-asiatica") -> ActionPlanResponse:
    return ActionPlanResponse(
        disease_id=disease_id,
        levels=[
            ActionPlanLevelResponse(
                level=ActionPlanLevelEnum.ESSENCIAL,
                actions=["Aplicar fungicida", "Monitorar lavoura"],
            ),
            ActionPlanLevelResponse(
                level=ActionPlanLevelEnum.CAMPO,
                actions=["Rotacionar culturas"],
            ),
        ],
        sources=[
            SourceResponse(
                id="src-1",
                name="EMBRAPA",
                detail="Fonte técnica",
                url="https://embrapa.br",
                display_order=0,
            )
        ],
    )


def _make_disease_dto(
    slug: str = "ferrugem-asiatica",
    name_pt: str = "Ferrugem Asiática",
    scientific_name: str | None = "Phakopsora pachyrhizi",
    severity_default: str = "alta",
    description_md: str | None = "Doença severa.",
):
    from app.domains.inference.repository import DiseaseDTO

    return DiseaseDTO(
        id=f"dto-{slug}",
        crop_id="soja-id",
        slug=slug,
        name_pt=name_pt,
        scientific_name=scientific_name,
        severity_default=severity_default,
        description_md=description_md,
        image_url=None,
    )


_MOCK_CATALOG = [
    _make_disease_dto(),
    _make_disease_dto(
        slug="mancha-alvo",
        name_pt="Mancha-Alvo",
        scientific_name="Corynespora cassiicola",
        severity_default="media",
    ),
    _make_disease_dto(
        slug="antracnose",
        name_pt="Antracnose",
        scientific_name="Colletotrichum truncatum",
        severity_default="media",
    ),
]


@pytest.fixture
def mock_inference_svc() -> MagicMock:
    svc = MagicMock()
    svc.predict.return_value = _make_inference_result()
    svc.disease_catalog = list(_MOCK_CATALOG)

    def _get_by_slug(slug: str):
        return next((d for d in _MOCK_CATALOG if d.slug == slug), None)

    svc.get_disease_by_slug.side_effect = _get_by_slug
    return svc


@pytest.fixture
def mock_action_plan_svc() -> AsyncMock:
    svc = AsyncMock()
    svc.get_by_disease.return_value = _make_action_plan_response()
    return svc


def _initial_state(
    user_text: str = "ola",
    image_filename: str | None = None,
    model_id: str = "ensemble",
) -> ChatState:
    return ChatState(
        messages=[HumanMessage(content=user_text)],
        current_user_id="user-uuid-1",
        image_filename=image_filename,
        model_id=model_id,
        last_diagnosis_id=None,
    )


def _ai_with_tool_call(tool_name: str, args: dict, call_id: str = "call-1") -> AIMessage:
    """Constrói um AIMessage com tool_call no formato esperado pelo ToolNode."""
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": tool_name,
                "args": args,
                "id": call_id,
                "type": "tool_call",
            }
        ],
    )


# ── Tools de apoio ────────────────────────────────────────────────────────────
#
# O ``build_graph`` nao monta mais tools sozinho — quem decide o conjunto ativo
# eh o ``tool_registry`` (ver ``tests/unit/chat/test_tool_registry.py`` e
# ``test_service_tool_registry.py``). Estes testes exercitam o GRAFO, entao
# montam aqui um trio minimo de tools sobre os services mockados. O
# comportamento de cada tool real e' testado em ``tests/unit/chat/tools/``.


def _graph_tools(inference_svc, action_plan_svc):
    """Trio minimo de tools pros testes de grafo (LLM -> ToolNode -> LLM)."""
    import json as _json

    from langchain_core.tools import tool

    @tool
    def analyze_image(image_filename: str, model_id: str) -> str:
        """Analisa uma imagem de folha de soja com o modelo indicado."""
        result = inference_svc.predict(model_id, image_filename)
        return _json.dumps(
            {
                "disease_name": result.disease_name,
                "disease_id": result.disease_id,
                "confidence": result.confidence,
            },
            ensure_ascii=False,
        )

    @tool
    async def get_action_plan(disease_id: str) -> str:
        """Plano de acao de uma doenca."""
        plan = await action_plan_svc.get_by_disease(disease_id)
        return f"Plano de acao para {plan.disease_id}"

    @tool
    def get_disease_info(disease_id: str) -> str:
        """Informacoes sobre uma doenca do catalogo."""
        disease = inference_svc.get_disease_by_slug(disease_id)
        return disease.name_pt if disease is not None else f"{disease_id} nao encontrada"

    return [analyze_image, get_action_plan, get_disease_info]


# ── Graph end-to-end com FakeLLM ──────────────────────────────────────────────


async def test_graph_no_tool_path(mock_inference_svc, mock_action_plan_svc):
    """LLM responde direto sem chamar tool — caminho mais simples."""
    fake_llm = FakeToolLLM(responses=[AIMessage(content="Olá! Como posso ajudar?")])
    graph = build_graph(
        _graph_tools(mock_inference_svc, mock_action_plan_svc), llm=fake_llm
    )

    result = await graph.ainvoke(_initial_state(user_text="oi"))

    # Última mensagem deve ser a resposta do LLM
    last = result["messages"][-1]
    assert isinstance(last, AIMessage)
    assert last.content == "Olá! Como posso ajudar?"

    mock_inference_svc.predict.assert_not_called()
    mock_action_plan_svc.get_by_disease.assert_not_called()


async def test_graph_includes_system_prompt_on_first_call(mock_inference_svc, mock_action_plan_svc):
    """Garante que o system prompt é injetado quando ausente."""
    captured = {}

    async def capturing_invoke(messages, *args, **kwargs):
        captured["messages"] = messages
        return AIMessage(content="ok")

    fake_llm = FakeToolLLM(responses=[AIMessage(content="ok")])
    bound = fake_llm.bind_tools(_graph_tools(mock_inference_svc, mock_action_plan_svc))
    # Patch ainvoke (object.__setattr__ pra contornar pydantic frozen)
    object.__setattr__(bound, "ainvoke", capturing_invoke)

    # Monta o grafo manualmente injetando o bound já capturado
    from langgraph.graph import END, START, StateGraph
    from langgraph.prebuilt import ToolNode, tools_condition

    from app.domains.chat.agent import ChatState, _make_llm_node

    tools = _graph_tools(mock_inference_svc, mock_action_plan_svc)
    wf: StateGraph = StateGraph(ChatState)
    wf.add_node("llm", _make_llm_node(bound))
    wf.add_node("tools", ToolNode(tools))
    wf.add_edge(START, "llm")
    wf.add_conditional_edges("llm", tools_condition, {"tools": "tools", "__end__": END})
    wf.add_edge("tools", "llm")
    graph = wf.compile()

    await graph.ainvoke(_initial_state(user_text="oi"))

    sent = captured["messages"]
    assert sent[0].content == SYSTEM_PROMPT


async def test_graph_analyze_image_path(mock_inference_svc, mock_action_plan_svc):
    """LLM chama analyze_image, tool retorna resultado, LLM finaliza."""
    fake_llm = FakeToolLLM(
        responses=[
            _ai_with_tool_call(
                "analyze_image",
                {"image_filename": "folha.jpg", "model_id": "ensemble"},
            ),
            AIMessage(content="Detectei Ferrugem Asiática com alta confiança."),
        ]
    )
    graph = build_graph(
        _graph_tools(mock_inference_svc, mock_action_plan_svc), llm=fake_llm
    )

    result = await graph.ainvoke(
        _initial_state(user_text="analisa minha folha", image_filename="folha.jpg")
    )

    last = result["messages"][-1]
    assert isinstance(last, AIMessage)
    assert "Ferrugem" in last.content
    mock_inference_svc.predict.assert_called_once_with("ensemble", "folha.jpg")


async def test_graph_get_disease_info_path(mock_inference_svc, mock_action_plan_svc):
    """LLM chama get_disease_info pra responder pergunta sobre doença."""
    fake_llm = FakeToolLLM(
        responses=[
            _ai_with_tool_call("get_disease_info", {"disease_id": "antracnose"}),
            AIMessage(content="Antracnose é causada por Colletotrichum truncatum."),
        ]
    )
    graph = build_graph(
        _graph_tools(mock_inference_svc, mock_action_plan_svc), llm=fake_llm
    )

    result = await graph.ainvoke(_initial_state(user_text="me fala da antracnose"))

    last = result["messages"][-1]
    assert "Antracnose" in last.content
    # Inference svc não deve ter sido chamado nesse path
    mock_inference_svc.predict.assert_not_called()


async def test_graph_get_action_plan_path(mock_inference_svc, mock_action_plan_svc):
    """LLM chama get_action_plan pra trazer recomendações."""
    fake_llm = FakeToolLLM(
        responses=[
            _ai_with_tool_call("get_action_plan", {"disease_id": "ferrugem-asiatica"}),
            AIMessage(content="Recomendo aplicar fungicida triazol + estrobilurina."),
        ]
    )
    graph = build_graph(
        _graph_tools(mock_inference_svc, mock_action_plan_svc), llm=fake_llm
    )

    result = await graph.ainvoke(_initial_state(user_text="o que fazer?"))

    last = result["messages"][-1]
    assert "fungicida" in last.content
    mock_action_plan_svc.get_by_disease.assert_awaited_once_with("ferrugem-asiatica")


async def test_graph_with_default_chatopenai(mock_inference_svc, mock_action_plan_svc, monkeypatch):
    """Quando llm não é passado, deve cair no get_chat_model da config (sem chamar de verdade)."""
    from app.domains.chat import agent as agent_mod

    sentinel_calls = {}

    class _SentinelLLM:
        def bind_tools(self, tools):
            return FakeToolLLM(responses=[AIMessage(content="fallback")])

    def _fake_get_chat_model(model_id, **kwargs):
        sentinel_calls["init_kwargs"] = {"model": model_id, **kwargs}
        return _SentinelLLM()

    monkeypatch.setattr(agent_mod, "get_chat_model", _fake_get_chat_model)

    graph = build_graph(_graph_tools(mock_inference_svc, mock_action_plan_svc))
    result = await graph.ainvoke(_initial_state(user_text="oi"))

    assert "model" in sentinel_calls["init_kwargs"]
    assert result["messages"][-1].content == "fallback"


# ── TCC-051: LLM model switching por plan_features ──────────────────────────


async def test_graph_uses_free_llm_model_when_plan_is_free(
    mock_inference_svc, mock_action_plan_svc, monkeypatch
):
    from app.domains.chat import agent as agent_mod
    from app.domains.subscriptions.features import FREE_FEATURES

    sentinel_calls = {}

    class _SentinelLLM:
        def bind_tools(self, tools):
            return FakeToolLLM(responses=[AIMessage(content="ok")])

    def _fake_get_chat_model(model_id, **kwargs):
        sentinel_calls["init_kwargs"] = {"model": model_id, **kwargs}
        return _SentinelLLM()

    monkeypatch.setattr(agent_mod, "get_chat_model", _fake_get_chat_model)

    build_graph(
        _graph_tools(mock_inference_svc, mock_action_plan_svc),
        plan_features=FREE_FEATURES,
    )
    assert sentinel_calls["init_kwargs"]["model"] == FREE_FEATURES.llm_model


async def test_graph_uses_pro_llm_model_when_plan_is_pro(
    mock_inference_svc, mock_action_plan_svc, monkeypatch
):
    from app.domains.chat import agent as agent_mod
    from app.domains.subscriptions.features import PRO_FEATURES

    sentinel_calls = {}

    class _SentinelLLM:
        def bind_tools(self, tools):
            return FakeToolLLM(responses=[AIMessage(content="ok")])

    def _fake_get_chat_model(model_id, **kwargs):
        sentinel_calls["init_kwargs"] = {"model": model_id, **kwargs}
        return _SentinelLLM()

    monkeypatch.setattr(agent_mod, "get_chat_model", _fake_get_chat_model)

    build_graph(
        _graph_tools(mock_inference_svc, mock_action_plan_svc),
        plan_features=PRO_FEATURES,
    )
    assert sentinel_calls["init_kwargs"]["model"] == PRO_FEATURES.llm_model


async def test_graph_falls_back_to_settings_when_no_plan_features(
    mock_inference_svc, mock_action_plan_svc, monkeypatch
):
    """Sem plan_features, usa settings.chat_model (back-compat)."""
    from app.domains.chat import agent as agent_mod

    sentinel_calls = {}

    class _SentinelLLM:
        def bind_tools(self, tools):
            return FakeToolLLM(responses=[AIMessage(content="ok")])

    def _fake_get_chat_model(model_id, **kwargs):
        sentinel_calls["init_kwargs"] = {"model": model_id, **kwargs}
        return _SentinelLLM()

    monkeypatch.setattr(agent_mod, "get_chat_model", _fake_get_chat_model)

    build_graph(
        _graph_tools(mock_inference_svc, mock_action_plan_svc), plan_features=None
    )
    assert sentinel_calls["init_kwargs"]["model"] is not None
