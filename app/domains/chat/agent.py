"""LangGraph agent para o Zé Praga — orquestra LLM + tools (Sprint A2 / A3).

A partir de TCC-041 as tools moram em ``app/domains/chat/tools/`` e sao
construidas via factories injetadas. Para retro-compat, o ``_build_tools``
legacy continua disponivel (consumido pelo ``ChatService`` atual e pelos
testes do PR #2), agora delegando para as factories novas com adapters.

Sprint A3 (TCC-051): ``build_graph`` aceita ``plan_features: PlanFeatures``
opcional pra escolher o LLM model (gpt-4o-mini vs gpt-4o) dinamicamente
por tier.

O grafo aceita ``tools: list[BaseTool] | None``:
    - se passado, usa direto;
    - senao, monta a lista legacy via ``_build_tools(inference_svc, action_plan_svc)``.

Estado:
    O ``ChatState`` legacy (com ``image_filename`` / ``last_diagnosis_id`` /
    ``model_id``) permanece neste modulo pra back-compat. O novo
    ``ChatState`` expandido (Sprint A2) mora em ``agent_state.py`` e sera
    consumido pelas tools novas quando o ``ChatService`` for migrado.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Annotated, Any, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import BaseTool, tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from app.config import settings
from app.core.llm import get_chat_model

if TYPE_CHECKING:
    from langchain_core.language_models import LanguageModelInput
    from langchain_core.runnables import Runnable
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.graph._node import StateNode
    from langgraph.graph.state import CompiledStateGraph

    from app.domains.action_plans.service import ActionPlanService
    from app.domains.inference.service import InferenceService
    from app.domains.subscriptions.features import PlanFeatures


SYSTEM_PROMPT = (
    "Você é o Zé Praga, assistente especializado em diagnóstico de doenças foliares de soja. "
    "Quando o usuário enviar uma imagem (image_filename presente no estado), use analyze_image. "
    "Quando perguntar sobre uma doença específica, use get_disease_info. "
    "Quando precisar de recomendações de manejo, use get_action_plan. "
    "Responda em português, com clareza e tom amigável."
)


class ChatState(TypedDict):
    """Estado legacy do grafo (mantido pra back-compat). `messages` mergeado via add_messages.

    O ChatState expandido (Sprint A2) mora em ``app/domains/chat/agent_state.py``.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    current_user_id: str
    image_filename: str | None
    model_id: str
    last_diagnosis_id: str | None


def _build_tools(
    inference_svc: InferenceService,
    action_plan_svc: ActionPlanService,
) -> list[Any]:
    """Cria as 3 tools legacy fechando sobre os services injetados.

    Mantido pra back-compat — o ``ChatService`` atual ainda chama
    ``build_graph(inference_svc, action_plan_svc, ...)``. Quando o service
    for migrado pra montar tools via ``build_tools(factories)`` do
    ``tool_registry``, este helper pode ser removido.
    """

    @tool
    def analyze_image(image_filename: str, model_id: str) -> str:
        """Analisa uma imagem de folha de soja usando o modelo CNN/ViT especificado.

        Args:
            image_filename: nome do arquivo da imagem (já enviado pelo usuário).
            model_id: identificador do modelo (resnet50, efficientnet, vit, ensemble).

        Returns:
            JSON-string com disease_name, disease_id, confidence, severity e top3.
        """
        result = inference_svc.predict(model_id, image_filename)
        payload = {
            "disease_name": result.disease_name,
            "disease_id": result.disease_id,
            "scientific_name": result.scientific_name,
            "confidence": result.confidence,
            "severity": str(result.severity),
            "description": result.description,
            "top3": [
                {
                    "rank": p.rank,
                    "disease_name": p.disease_name,
                    "disease_id": p.disease_id,
                    "confidence": p.confidence,
                    "severity": p.severity,
                }
                for p in result.top3
            ],
        }
        return json.dumps(payload, ensure_ascii=False)

    @tool
    async def get_action_plan(disease_id: str) -> str:
        """Busca o plano de ação para uma doença identificada por disease_id.

        Retorna texto agregando níveis (essencial, campo, especialista) e fontes.
        """
        try:
            plan = await action_plan_svc.get_by_disease(disease_id)
        except Exception as exc:  # noqa: BLE001 — tool sempre retorna string
            return f"Plano de ação indisponível para {disease_id}: {exc}"

        lines: list[str] = [f"Plano de ação para {plan.disease_id}:"]
        for level in plan.levels:
            lines.append(f"\n[{level.level.upper()}]")
            for action in level.actions:
                lines.append(f"- {action}")
        if plan.sources:
            lines.append("\nFontes:")
            for src in plan.sources:
                suffix = f" ({src.url})" if src.url else ""
                lines.append(f"- {src.name}: {src.detail}{suffix}")
        return "\n".join(lines)

    @tool
    def get_disease_info(disease_id: str) -> str:
        """Retorna informações estruturadas sobre uma doença a partir do catálogo interno.

        Args:
            disease_id: identificador (slug) da doença (ex: ferrugem-asiatica, mancha-alvo).
        """
        # Consulta o catálogo via service (que recebeu DiseaseDTOs do DB via DI).
        disease = inference_svc.get_disease_by_slug(disease_id)
        if disease is not None:
            payload = {
                "id": disease.slug,
                "name": disease.name_pt,
                "scientific_name": disease.scientific_name,
                "severity": str(disease.severity_default),
                "description": disease.description_md,
            }
            return json.dumps(payload, ensure_ascii=False)

        valid_ids = [d.slug for d in inference_svc.disease_catalog]
        return (
            f"Doença '{disease_id}' não encontrada no catálogo. "
            f"IDs válidos: {', '.join(valid_ids)}."
        )

    return [analyze_image, get_action_plan, get_disease_info]


def _make_llm_node(
    llm_with_tools: Runnable[LanguageModelInput, BaseMessage],
) -> StateNode[ChatState]:
    """Cria o nó do LLM que injeta o SystemPrompt no início, se ausente."""

    async def llm_node(state: ChatState) -> dict[str, list[BaseMessage]]:
        messages = state["messages"]
        # Garante system prompt na primeira chamada do turno.
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SYSTEM_PROMPT), *messages]
        response = await llm_with_tools.ainvoke(messages)
        return {"messages": [response]}

    return llm_node


def build_graph(
    inference_svc: InferenceService | None = None,
    action_plan_svc: ActionPlanService | None = None,
    llm: BaseChatModel | None = None,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    tools: list[BaseTool] | None = None,
    state_schema: type | None = None,
    summarize_llm: BaseChatModel | None = None,
    plan_features: PlanFeatures | None = None,
) -> CompiledStateGraph[Any]:
    """Monta e compila o grafo do Zé Praga.

    Modo legado (TCC-009): passa ``inference_svc`` + ``action_plan_svc`` e o
    helper monta as 3 tools (analyze_image, get_disease_info, get_action_plan).

    Modo novo (TCC-041): passa ``tools`` pre-construidas via factories. Quando
    ``tools`` eh dado, ``inference_svc``/``action_plan_svc`` viram opcionais.

    Em Sprint A2.5 (TCC-047) introduzimos ``maybe_summarize_node`` ANTES do
    LLM — comprime historico quando ele cresce alem de 20 mensagens.

    Sprint A3 (TCC-051): se ``plan_features`` for passado e ``llm`` for None,
    instancia ChatOpenAI com ``plan_features.llm_model`` em vez de
    ``settings.openai_model``. Permite Free=gpt-4o-mini, Pro/Enterprise=gpt-4o
    sem reconfigurar settings.

    Args:
        inference_svc: serviço de inferência (mock CNN/ViT). Usado so quando
            ``tools`` eh None.
        action_plan_svc: serviço de planos de ação. Usado so quando ``tools``
            eh None.
        llm: chat model opcional (default: ChatOpenAI da config). Útil pra
            injetar FakeLLM nos testes.
        checkpointer: checkpointer opcional do langgraph (ex: MemorySaver).
        tools: lista de tools pre-construidas (override do helper legacy).
        state_schema: tipo de estado (default: ChatState legacy). Pra usar o
            ChatState expandido novo, passe ``agent_state.ChatState``.
        summarize_llm: LLM dedicado pra rolling summary. Quando ``None``
            usa o mesmo ``llm`` do agente principal.
        plan_features: PlanFeatures opcional pra escolher o LLM model dinamico
            (Free/Pro/Enterprise). Quando None, usa ``settings.openai_model``.

    Returns:
        Grafo compilado pronto pra `.ainvoke()` / `.astream_events()`.
    """
    if tools is None:
        if inference_svc is None or action_plan_svc is None:
            raise ValueError(
                "build_graph requer 'tools' OU (inference_svc + action_plan_svc)"
            )
        tools = _build_tools(inference_svc, action_plan_svc)

    if llm is None:
        model_id = (
            plan_features.llm_model if plan_features is not None else settings.chat_model
        )
        llm = get_chat_model(model_id)

    llm_with_tools = llm.bind_tools(tools)

    schema = state_schema or ChatState
    workflow: StateGraph[Any] = StateGraph(schema)

    # Sprint A2.5: rolling summary antes do LLM pra controlar custo/contexto.
    from functools import partial

    from app.domains.chat.nodes import maybe_summarize_node

    summary_llm = summarize_llm or llm
    workflow.add_node(
        "maybe_summarize",
        partial(maybe_summarize_node, llm=summary_llm),
    )
    workflow.add_node("llm", _make_llm_node(llm_with_tools))
    workflow.add_node("tools", ToolNode(tools))

    workflow.add_edge(START, "maybe_summarize")
    workflow.add_edge("maybe_summarize", "llm")
    workflow.add_conditional_edges(
        "llm",
        tools_condition,
        {"tools": "tools", "__end__": END},
    )
    workflow.add_edge("tools", "llm")

    return workflow.compile(checkpointer=checkpointer)
