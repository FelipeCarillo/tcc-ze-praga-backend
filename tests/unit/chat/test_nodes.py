"""Testes do maybe_summarize_node (TCC-047)."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)

from app.domains.chat.nodes import maybe_summarize_node


def _msg(cls, content: str, msg_id: str):
    """Cria mensagem com id explicito pro RemoveMessage funcionar."""
    return cls(content=content, id=msg_id)


async def test_noop_under_threshold():
    """Com <= 20 mensagens, node nao chama LLM e retorna dict vazio."""
    fake_llm = AsyncMock()
    state = {
        "messages": [
            _msg(HumanMessage, f"msg-{i}", f"id-{i}") for i in range(15)
        ]
    }

    update = await maybe_summarize_node(state, llm=fake_llm)

    assert update == {}
    fake_llm.ainvoke.assert_not_called()


async def test_noop_at_exact_threshold():
    """Exatamente 20 mensagens — ainda nao compressa (threshold > 20)."""
    fake_llm = AsyncMock()
    state = {
        "messages": [
            _msg(HumanMessage, f"msg-{i}", f"id-{i}") for i in range(20)
        ]
    }

    update = await maybe_summarize_node(state, llm=fake_llm)

    assert update == {}
    fake_llm.ainvoke.assert_not_called()


async def test_compresses_above_threshold():
    """Com 25 mensagens, compressa: gera summary, emite RemoveMessage para
    todas as 25 (com id) + summary + 10 keep."""
    fake_summary = AIMessage(content="Resumo da conversa anterior.")
    fake_llm = AsyncMock()
    fake_llm.ainvoke.return_value = fake_summary

    messages = [
        _msg(HumanMessage, f"msg-{i}", f"id-{i}") for i in range(25)
    ]
    state = {"messages": messages}

    update = await maybe_summarize_node(state, llm=fake_llm)

    assert "messages" in update
    out = update["messages"]

    # 25 RemoveMessages + 1 summary + 10 keep = 36
    remove_count = sum(1 for m in out if isinstance(m, RemoveMessage))
    system_count = sum(1 for m in out if isinstance(m, SystemMessage))
    human_count = sum(1 for m in out if isinstance(m, HumanMessage))

    assert remove_count == 25
    assert system_count == 1  # apenas o summary
    assert human_count == 10  # ultimas 10 keep

    # Summary deve conter o texto gerado pelo LLM
    summary_msg = next(m for m in out if isinstance(m, SystemMessage))
    assert "Resumo da conversa anterior." in summary_msg.content
    assert "Conversa anterior resumida" in summary_msg.content

    fake_llm.ainvoke.assert_awaited_once()


async def test_llm_receives_messages_to_compress():
    """LLM eh chamado com o prompt + as primeiras N-10 mensagens."""
    fake_summary = AIMessage(content="resumo")
    fake_llm = AsyncMock()
    fake_llm.ainvoke.return_value = fake_summary

    messages = [
        _msg(HumanMessage, f"old-{i}", f"id-{i}") for i in range(15)
    ] + [
        _msg(AIMessage, f"new-{i}", f"id-new-{i}") for i in range(10)
    ]
    # Total: 25 messages. Primeiras 15 vao compressas, ultimas 10 keep.
    state = {"messages": messages}

    await maybe_summarize_node(state, llm=fake_llm)

    call_args = fake_llm.ainvoke.call_args.args[0]
    # call_args = [SystemMessage(prompt), ...primeiras 15...]
    assert isinstance(call_args[0], SystemMessage)
    assert "Resuma" in call_args[0].content
    assert len(call_args) == 16  # 1 prompt + 15 mensagens pra compressar


async def test_handles_content_as_list_parts():
    """Quando o LLM retorna content como list de parts, ainda gera summary."""
    fake_summary = AIMessage(
        content=[{"type": "text", "text": "part 1 "}, {"type": "text", "text": "part 2"}]
    )
    fake_llm = AsyncMock()
    fake_llm.ainvoke.return_value = fake_summary

    messages = [
        _msg(HumanMessage, f"m-{i}", f"id-{i}") for i in range(25)
    ]
    state = {"messages": messages}

    update = await maybe_summarize_node(state, llm=fake_llm)
    summary_msg = next(
        m for m in update["messages"] if isinstance(m, SystemMessage)
    )
    assert "part 1" in summary_msg.content
    assert "part 2" in summary_msg.content


async def test_messages_without_id_are_not_removed():
    """Mensagens sem id nao recebem RemoveMessage (defensivo)."""
    fake_summary = AIMessage(content="resumo")
    fake_llm = AsyncMock()
    fake_llm.ainvoke.return_value = fake_summary

    # 25 mensagens, primeira sem id
    messages = [HumanMessage(content="no-id")]
    messages += [
        _msg(HumanMessage, f"m-{i}", f"id-{i}") for i in range(24)
    ]
    state = {"messages": messages}

    update = await maybe_summarize_node(state, llm=fake_llm)
    remove_count = sum(
        1 for m in update["messages"] if isinstance(m, RemoveMessage)
    )
    assert remove_count == 24  # so as que tem id
