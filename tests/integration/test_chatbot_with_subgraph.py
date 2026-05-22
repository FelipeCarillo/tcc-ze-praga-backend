"""Integration tests: chatbot_graph + tool_registry + diagnosis_graph (TCC-043).

Estes testes exercitam o caminho completo Sprint A2:
    FakeLLM -> deep_diagnose tool -> diagnosis_graph -> persist -> consolida

Em todos os cenarios os 3 services do sub-grafo sao mockados, garantindo
que o teste seja determinista e nao bata no DB nem na OpenAI.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage, HumanMessage

from app.domains.action_plans.schemas import (
    ActionPlanLevelResponse,
    ActionPlanResponse,
    SourceResponse,
)
from app.domains.chat.agent import build_graph
from app.domains.chat.agent_state import ChatState as RichChatState
from app.domains.chat.agent_state import UploadedFileDTO
from app.domains.chat.tool_registry import build_tools
from app.domains.chat.tools import (
    build_deep_diagnose_tool,
    build_get_action_plan_tool,
    build_search_my_diagnoses_tool,
)
from app.domains.diagnoses.schemas import DiagnosisResponse, Top3PredictionSchema
from app.domains.diagnosis_graph.graph import build_diagnosis_graph
from app.domains.inference.schemas import InferenceResult
from app.shared.enums import ActionPlanLevelEnum, SeverityEnum
from tests.conftest import NOW


# ── FakeLLM with bind_tools no-op ─────────────────────────────────────────────


class FakeToolLLM(FakeMessagesListChatModel):
    """FakeMessagesListChatModel + bind_tools no-op."""

    def bind_tools(self, tools, **kwargs):  # type: ignore[override]
        return self


def _ai_with_tool_call(
    tool_name: str, args: dict, call_id: str = "call-1"
) -> AIMessage:
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


# ── Helpers que constroem o sub-grafo real com services mockados ─────────────


def _inference_result(disease: str = "Ferrugem Asiática") -> InferenceResult:
    return InferenceResult(
        disease_id="ferrugem-asiatica",
        disease_name=disease,
        scientific_name="Phakopsora pachyrhizi",
        severity=SeverityEnum.ALTA,
        description="Doença severa.",
        confidence=0.92,
        model_id="ensemble",
        image_name="leaf.jpg",
        top3=[
            Top3PredictionSchema(
                rank=1,
                disease_name=disease,
                disease_id="ferrugem-asiatica",
                scientific_name="Phakopsora pachyrhizi",
                confidence=0.92,
                severity="alta",
            )
        ],
    )


def _plan() -> ActionPlanResponse:
    return ActionPlanResponse(
        disease_id="ferrugem-asiatica",
        levels=[
            ActionPlanLevelResponse(
                level=ActionPlanLevelEnum.ESSENCIAL,
                actions=["Aplicar fungicida triazol"],
            )
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


def _diag_response(diag_id: str = "diag-1") -> DiagnosisResponse:
    return DiagnosisResponse(
        id=diag_id,
        disease_name="Ferrugem Asiática",
        disease_id="ferrugem-asiatica",
        scientific_name="Phakopsora pachyrhizi",
        confidence=0.92,
        severity="alta",
        description=None,
        model_used="ensemble",
        image_url=None,
        image_name="leaf.jpg",
        created_at=NOW,
        top3=[],
    )


@pytest.fixture
def services_triplet():
    """3 services mockados que o sub-grafo precisa."""
    inference = MagicMock()
    inference.predict.return_value = _inference_result()

    action_plan = AsyncMock()
    action_plan.get_by_disease.return_value = _plan()

    diagnosis = AsyncMock()

    return inference, action_plan, diagnosis


@pytest.fixture
def diagnosis_graph_factory(services_triplet):
    """Factory cacheada que retorna o sub-grafo real com services mockados."""
    inference, action_plan, diagnosis = services_triplet
    _cache: dict[str, object] = {}

    def _factory(crop_id: str):
        if crop_id not in _cache:
            _cache[crop_id] = build_diagnosis_graph(
                inference, action_plan, diagnosis
            )
        return _cache[crop_id]

    return _factory


# ── 1) End-to-end: FakeLLM chama deep_diagnose -> sub-grafo -> persiste ──────


async def test_chatbot_with_subgraph_persists_diagnosis(
    services_triplet, diagnosis_graph_factory
):
    """LLM emite tool_call -> deep_diagnose roda sub-grafo -> diagnosis_svc.create
    eh invocado -> LLM responde consolidando."""
    inference, action_plan, diagnosis = services_triplet
    diagnosis.create.return_value = _diag_response("diag-final-1")

    deep_diagnose = build_deep_diagnose_tool(diagnosis_graph_factory)
    fake_llm = FakeToolLLM(
        responses=[
            _ai_with_tool_call("deep_diagnose", {}),
            AIMessage(
                content="Detectei Ferrugem Asiática com alta confianca."
            ),
        ]
    )

    graph = build_graph(
        tools=[deep_diagnose], llm=fake_llm, state_schema=RichChatState
    )

    file_dto = UploadedFileDTO(
        id="img-1",
        original_name="leaf.jpg",
        mime="image/jpeg",
        storage_key="uploads/u1/leaf.jpg",
        size_bytes=1024,
    )

    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="analisa essa folha")],
            "current_user_id": "user-uuid-1",
            "selected_model": "ensemble",
            "uploaded_files": [file_dto],
        }
    )

    # LLM finalizou
    last = result["messages"][-1]
    assert isinstance(last, AIMessage)
    assert "Ferrugem" in last.content

    # Sub-grafo invocou inference + action_plan + diagnosis_svc.create
    inference.predict.assert_called_once()
    action_plan.get_by_disease.assert_awaited_once_with("ferrugem-asiatica")
    diagnosis.create.assert_awaited_once()
    # User id veio do state via InjectedState
    assert diagnosis.create.await_args.args[0] == "user-uuid-1"


# ── 2) Batch: 3 imagens em 1 tool_call -> 3 diagnoses ────────────────────────


async def test_chatbot_subgraph_batch_three_images(
    services_triplet, diagnosis_graph_factory
):
    inference, action_plan, diagnosis = services_triplet
    diagnosis.create.side_effect = [
        _diag_response(f"diag-{i}") for i in range(3)
    ]

    deep_diagnose = build_deep_diagnose_tool(diagnosis_graph_factory)
    fake_llm = FakeToolLLM(
        responses=[
            _ai_with_tool_call("deep_diagnose", {}),
            AIMessage(content="Analisei as 3 imagens."),
        ]
    )

    graph = build_graph(
        tools=[deep_diagnose], llm=fake_llm, state_schema=RichChatState
    )
    files = [
        UploadedFileDTO(
            id=f"img-{i}",
            original_name=f"leaf-{i}.jpg",
            mime="image/jpeg",
            storage_key=f"u1/{i}.jpg",
            size_bytes=1024,
        )
        for i in range(3)
    ]

    await graph.ainvoke(
        {
            "messages": [HumanMessage(content="analisa as 3")],
            "current_user_id": "user-uuid-1",
            "selected_model": "ensemble",
            "uploaded_files": files,
        }
    )

    assert diagnosis.create.await_count == 3
    assert inference.predict.call_count == 3


# ── 3) Filter: image_ids especifica subset ───────────────────────────────────


async def test_chatbot_subgraph_filters_image_ids(
    services_triplet, diagnosis_graph_factory
):
    inference, action_plan, diagnosis = services_triplet
    diagnosis.create.return_value = _diag_response("diag-only")

    deep_diagnose = build_deep_diagnose_tool(diagnosis_graph_factory)
    fake_llm = FakeToolLLM(
        responses=[
            _ai_with_tool_call(
                "deep_diagnose", {"image_ids": ["img-b"]}
            ),
            AIMessage(content="Analisei so a 2 foto."),
        ]
    )

    graph = build_graph(
        tools=[deep_diagnose], llm=fake_llm, state_schema=RichChatState
    )

    files = [
        UploadedFileDTO(
            id=fid,
            original_name=f"{fid}.jpg",
            mime="image/jpeg",
            storage_key=f"u1/{fid}.jpg",
            size_bytes=1024,
        )
        for fid in ("img-a", "img-b", "img-c")
    ]

    await graph.ainvoke(
        {
            "messages": [HumanMessage(content="analisa so a do meio")],
            "current_user_id": "user-uuid-1",
            "selected_model": "ensemble",
            "uploaded_files": files,
        }
    )

    inference.predict.assert_called_once()
    diagnosis.create.assert_awaited_once()


# ── 4) Tool registry: monta tools ativas via build_tools ─────────────────────


def test_build_tools_with_factories_returns_4_tools(
    diagnosis_graph_factory, services_triplet
):
    """``build_tools`` deve retornar as 4 tools default ativas pra Free tier."""
    _inference, action_plan, _diag = services_triplet

    factories = {
        "deep_diagnose": lambda: build_deep_diagnose_tool(
            diagnosis_graph_factory
        ),
        "get_disease_info": lambda: _stub_tool("get_disease_info"),
        "get_action_plan": lambda: build_get_action_plan_tool(action_plan),
        "search_my_diagnoses": lambda: build_search_my_diagnoses_tool(),
    }

    tools = build_tools(factories, {"tier_name": "free"})
    names = {t.name for t in tools}
    assert names == {
        "deep_diagnose",
        "get_disease_info",
        "get_action_plan",
        "search_my_diagnoses",
    }


async def test_chatbot_uses_only_active_tools_from_registry(
    services_triplet, diagnosis_graph_factory
):
    """A composicao via registry + build_tools deve produzir tools utilizaveis
    pelo agente, e o LLM consegue invocar uma delas."""
    inference, action_plan, diagnosis = services_triplet
    diagnosis.create.return_value = _diag_response()

    factories = {
        "deep_diagnose": lambda: build_deep_diagnose_tool(
            diagnosis_graph_factory
        ),
        "get_disease_info": lambda: _stub_tool("get_disease_info"),
        "get_action_plan": lambda: build_get_action_plan_tool(action_plan),
        "search_my_diagnoses": lambda: build_search_my_diagnoses_tool(),
    }
    active_tools = build_tools(factories, {"tier_name": "free"})

    fake_llm = FakeToolLLM(
        responses=[
            _ai_with_tool_call("deep_diagnose", {}),
            AIMessage(content="Ferrugem detectada."),
        ]
    )
    graph = build_graph(
        tools=active_tools, llm=fake_llm, state_schema=RichChatState
    )

    file_dto = UploadedFileDTO(
        id="img-1",
        original_name="leaf.jpg",
        mime="image/jpeg",
        storage_key="u1/x.jpg",
        size_bytes=1024,
    )
    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="diagnostica")],
            "current_user_id": "user-uuid-1",
            "selected_model": "ensemble",
            "uploaded_files": [file_dto],
        }
    )

    last = result["messages"][-1]
    assert isinstance(last, AIMessage)
    assert "Ferrugem" in last.content
    diagnosis.create.assert_awaited_once()


# ── Helper: stub tool for get_disease_info (avoids DB dependency) ────────────


def _stub_tool(name: str):
    """Cria uma stub tool sem dependencia de DB (uso so' em registry tests)."""
    from langchain_core.tools import tool as _tool

    @_tool
    async def _stub(*, query: str = "") -> str:  # noqa: ARG001
        """Stub tool."""
        return "{}"

    _stub.name = name
    return _stub
