"""LangGraph agent para o Zé Praga — orquestra LLM + tools (Sprint A2 / A3).

As tools moram em ``app/domains/chat/tools/`` e sao construidas via factories
injetadas; quem decide o conjunto ativo eh o ``tool_registry``, chamado pelo
``ChatService``. Este modulo so' monta e compila o grafo.

Sprint A3 (TCC-051): ``build_graph`` aceita ``plan_features: PlanFeatures``
opcional pra escolher o LLM model (gpt-4o-mini vs gpt-4o) dinamicamente
por tier.

Estado:
    O ``ChatState`` deste modulo (com ``image_filename`` / ``last_diagnosis_id``
    / ``model_id``) eh o schema minimo usado por testes de grafo. Em producao o
    ``ChatService`` passa o ``ChatState`` expandido de ``agent_state.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import BaseTool
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

    from app.domains.subscriptions.features import PlanFeatures


SYSTEM_PROMPT = """Você é o Zé Praga, assistente especializado em diagnóstico
de doenças foliares de soja.

FLUXO COM IMAGEM:
1. Chame SEMPRE inspect_image primeiro.
2. Se vier is_analyzable_plant=false, NÃO diagnostique: explique com gentileza que
   você só analisa fotos de plantas/folhas de cultivo e peça uma foto da folha
   afetada.
3. Se vier is_analyzable_plant=true, diagnostique:
   - UMA imagem no turno: use analyze_image.
   - VÁRIAS imagens no turno: use deep_diagnose (processa o lote de uma vez).
   Nunca chame as duas para a mesma imagem.
4. Explique o resultado de forma amigável e use get_action_plan quando fizer
   sentido trazer recomendações de manejo.

OUTRAS FERRAMENTAS:
- get_disease_info: perguntas sobre uma doença específica do catálogo.
- search_my_diagnoses: quando o usuário se referir a diagnósticos passados dele
  ("o que deu na semana passada?", "já tive isso antes?").
- compare_diagnoses: quando o usuário quiser confrontar modelos diferentes na
  mesma imagem.
- search_web / search_scientific: só quando o catálogo interno não bastar; cite
  as fontes que voltarem.

Nem toda ferramenta está disponível em todo plano — se uma não estiver na sua
lista, siga sem ela e não mencione que ela existe. Se um resultado de tool
trouxer "note" ou "model_downgraded_to", avise o usuário em uma frase curta.

Responda em português, com clareza, usando Markdown quando ajudar na leitura."""


class ChatState(TypedDict):
    """Estado legacy do grafo (mantido pra back-compat). `messages` mergeado via add_messages.

    O ChatState expandido (Sprint A2) mora em ``app/domains/chat/agent_state.py``.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    current_user_id: str
    image_filename: str | None
    model_id: str
    last_diagnosis_id: str | None


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
    tools: list[BaseTool],
    llm: BaseChatModel | None = None,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    state_schema: type | None = None,
    summarize_llm: BaseChatModel | None = None,
    plan_features: PlanFeatures | None = None,
) -> CompiledStateGraph[Any]:
    """Monta e compila o grafo do Zé Praga.

    As tools sao sempre montadas fora — em producao pelo ``tool_registry``, via
    ``ChatService._build_tool_factories``, que filtra por plano. O helper legado
    que montava 3 tools fixas aqui dentro foi removido quando o ``ChatService``
    passou a usar o registry: nao tinha mais chamador.

    Em Sprint A2.5 (TCC-047) introduzimos ``maybe_summarize_node`` ANTES do
    LLM — comprime historico quando ele cresce alem de 20 mensagens.

    Sprint A3 (TCC-051): se ``plan_features`` for passado e ``llm`` for None,
    instancia o chat model com ``plan_features.llm_model`` em vez de
    ``settings.chat_model``. Permite Free=gpt-4o-mini, Pro/Enterprise=gpt-4o
    sem reconfigurar settings.

    Args:
        tools: tools ja construidas, a serem bindadas no LLM.
        llm: chat model opcional (default: o da config). Útil pra injetar
            FakeLLM nos testes.
        checkpointer: checkpointer opcional do langgraph (ex: MemorySaver).
        state_schema: tipo de estado (default: ``ChatState`` deste modulo). Pra
            usar o ChatState expandido, passe ``agent_state.ChatState``.
        summarize_llm: LLM dedicado pra rolling summary. Quando ``None``
            usa o mesmo ``llm`` do agente principal.
        plan_features: PlanFeatures opcional pra escolher o LLM model dinamico
            (Free/Pro/Enterprise). Quando None, usa ``settings.chat_model``.

    Returns:
        Grafo compilado pronto pra `.ainvoke()` / `.astream_events()`.
    """
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
