"""Testes da tool ``ask_user`` (TCC-057).

Cobre:
- Disparo de ``GraphInterrupt`` quando a tool eh executada num grafo real
  com ``MemorySaver``.
- Snapshot do checkpointer contem o payload do interrupt.
- Resume via ``Command(resume=...)`` continua a execucao e a tool retorna a
  string esperada.
- Forma do payload (kind, question, response_kind, options, asked_at).
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command

from app.domains.chat.agent_state import ChatState
from app.domains.chat.tools.ask_user import build_ask_user_tool


def _build_test_graph(checkpointer: MemorySaver):
    """Mini grafo: START -> ask_user_tool_node -> END."""
    tool = build_ask_user_tool()
    workflow: StateGraph = StateGraph(ChatState)
    workflow.add_node("ask_user_tool", ToolNode([tool]))
    workflow.add_edge(START, "ask_user_tool")
    workflow.add_edge("ask_user_tool", END)
    return workflow.compile(checkpointer=checkpointer), tool


def _tool_call_message(tool_name: str, args: dict, call_id: str = "call-1"):
    """Cria um AIMessage que invoca a tool — formato esperado pelo ToolNode."""
    from langchain_core.messages import AIMessage

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


def test_build_ask_user_tool_has_expected_metadata() -> None:
    tool = build_ask_user_tool()
    assert tool.name == "ask_user"
    # Schema deve listar os 3 args do LLM (question, response_kind, options).
    # state aparece no args_schema bruto (Pydantic) mas o LangChain o oculta
    # via InjectedState quando bindado ao LLM — testado nos integration tests
    # do ChatService.
    schema = tool.args_schema.model_json_schema()
    props = schema["properties"]
    assert "question" in props
    assert "response_kind" in props
    assert "options" in props
    # Doc da tool deve mencionar HITL / interrupt na descricao pro LLM
    assert tool.description
    assert "usuario" in tool.description.lower()


async def test_ask_user_raises_interrupt_on_first_invocation() -> None:
    checkpointer = MemorySaver()
    graph, _ = _build_test_graph(checkpointer)
    config = {"configurable": {"thread_id": "thread-1"}}

    initial_state = {
        "messages": [
            _tool_call_message(
                "ask_user",
                {
                    "question": "Qual cultivo?",
                    "response_kind": "choice",
                    "options": ["soja", "milho"],
                },
            )
        ],
        "current_user_id": "u-1",
        "current_session_id": "s-1",
    }

    # interrupt() pausa o grafo — astream nao lanca exception ate o end de
    # cada step. Em vez disso o snapshot fica com tasks pendentes com
    # interrupts.
    await graph.ainvoke(initial_state, config=config)
    snapshot = await graph.aget_state(config)
    assert snapshot.tasks
    interrupts = [i for task in snapshot.tasks for i in task.interrupts]
    assert len(interrupts) == 1
    payload = interrupts[0].value
    assert payload["kind"] == "ask_user"
    assert payload["question"] == "Qual cultivo?"
    assert payload["response_kind"] == "choice"
    assert payload["options"] == ["soja", "milho"]
    assert "asked_at" in payload


async def test_ask_user_resume_returns_user_response() -> None:
    checkpointer = MemorySaver()
    graph, _ = _build_test_graph(checkpointer)
    config = {"configurable": {"thread_id": "thread-2"}}

    initial_state = {
        "messages": [
            _tool_call_message(
                "ask_user",
                {
                    "question": "Confirma o plano?",
                    "response_kind": "boolean",
                },
            )
        ],
        "current_user_id": "u-1",
        "current_session_id": "s-2",
    }

    await graph.ainvoke(initial_state, config=config)
    # Confirma interrupt pendente
    snapshot = await graph.aget_state(config)
    assert snapshot.tasks and snapshot.tasks[0].interrupts

    # Resume com a resposta
    final = await graph.ainvoke(Command(resume="sim"), config=config)
    messages = final["messages"]
    # ToolNode anexa um ToolMessage com o output da tool
    from langchain_core.messages import ToolMessage

    tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
    assert tool_msgs
    assert "Usuario respondeu: sim" in tool_msgs[-1].content


async def test_ask_user_default_response_kind_is_text() -> None:
    checkpointer = MemorySaver()
    graph, _ = _build_test_graph(checkpointer)
    config = {"configurable": {"thread_id": "thread-3"}}

    initial_state = {
        "messages": [
            _tool_call_message(
                "ask_user",
                {"question": "Algo mais?"},
            )
        ],
        "current_user_id": "u-1",
        "current_session_id": "s-3",
    }
    await graph.ainvoke(initial_state, config=config)
    snapshot = await graph.aget_state(config)
    payload = snapshot.tasks[0].interrupts[0].value
    assert payload["response_kind"] == "text"
    assert payload["options"] is None


async def test_ask_user_state_param_is_optional_and_ignored() -> None:
    """Sanity check: state injetado nao quebra mesmo vazio."""
    checkpointer = MemorySaver()
    graph, _ = _build_test_graph(checkpointer)
    config = {"configurable": {"thread_id": "thread-4"}}

    await graph.ainvoke(
        {
            "messages": [
                _tool_call_message(
                    "ask_user",
                    {"question": "?"},
                )
            ],
        },
        config=config,
    )
    snapshot = await graph.aget_state(config)
    assert snapshot.tasks[0].interrupts
