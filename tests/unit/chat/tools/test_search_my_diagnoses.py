"""Testes da tool search_my_diagnoses semantica (TCC-046)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

from app.domains.chat.tools.search_my_diagnoses import (
    build_search_my_diagnoses_tool,
)


async def test_returns_empty_when_no_user_id_in_state():
    """Sem current_user_id no state, tool retorna [] sem chamar o Store."""
    tool = build_search_my_diagnoses_tool(store_factory=AsyncMock())
    raw = await tool.ainvoke(
        {"query": "ferrugem", "limit": 5, "state": {}}
    )
    assert json.loads(raw) == []


async def test_returns_empty_when_no_store_factory():
    """Sem store_factory injetado, tool retorna [] (back-compat)."""
    tool = build_search_my_diagnoses_tool()
    raw = await tool.ainvoke(
        {
            "query": "ferrugem",
            "limit": 5,
            "state": {"current_user_id": "user-1"},
        }
    )
    assert json.loads(raw) == []


async def test_calls_store_asearch_with_correct_namespace():
    fake_store = MagicMock()
    fake_store.asearch = AsyncMock(return_value=[])
    store_factory = AsyncMock(return_value=fake_store)

    tool = build_search_my_diagnoses_tool(store_factory=store_factory)
    await tool.ainvoke(
        {
            "query": "ferrugem na soja",
            "limit": 3,
            "state": {"current_user_id": "user-1"},
        }
    )

    fake_store.asearch.assert_awaited_once()
    args = fake_store.asearch.call_args
    assert args.args[0] == ("user", "user-1", "diagnoses")
    assert args.kwargs["query"] == "ferrugem na soja"
    assert args.kwargs["limit"] == 3


async def test_returns_results_as_json_payload():
    fake_item_1 = MagicMock()
    fake_item_1.value = {
        "summary_text": "Diagnostico de Ferrugem",
        "diagnosis_id": "diag-1",
        "disease_id": "ferrugem-asiatica",
        "confidence": 0.91,
    }
    fake_item_2 = MagicMock()
    fake_item_2.value = {
        "summary_text": "Diagnostico de Mancha",
        "diagnosis_id": "diag-2",
        "disease_id": "mancha-alvo",
        "confidence": 0.75,
    }
    fake_store = MagicMock()
    fake_store.asearch = AsyncMock(return_value=[fake_item_1, fake_item_2])
    store_factory = AsyncMock(return_value=fake_store)

    tool = build_search_my_diagnoses_tool(store_factory=store_factory)
    raw = await tool.ainvoke(
        {
            "query": "ferrugem",
            "limit": 5,
            "state": {"current_user_id": "user-1"},
        }
    )

    parsed = json.loads(raw)
    assert len(parsed) == 2
    assert parsed[0]["diagnosis_id"] == "diag-1"
    assert parsed[1]["diagnosis_id"] == "diag-2"


async def test_swallows_store_errors_returns_empty():
    """Erros do Store nao explodem — tool retorna [] e loga."""
    fake_store = MagicMock()
    fake_store.asearch = AsyncMock(side_effect=RuntimeError("offline"))
    store_factory = AsyncMock(return_value=fake_store)

    tool = build_search_my_diagnoses_tool(store_factory=store_factory)
    raw = await tool.ainvoke(
        {
            "query": "ferrugem",
            "limit": 5,
            "state": {"current_user_id": "user-1"},
        }
    )

    assert json.loads(raw) == []


async def test_default_limit_is_5():
    fake_store = MagicMock()
    fake_store.asearch = AsyncMock(return_value=[])
    store_factory = AsyncMock(return_value=fake_store)

    tool = build_search_my_diagnoses_tool(store_factory=store_factory)
    await tool.ainvoke(
        {"query": "qualquer", "state": {"current_user_id": "user-1"}}
    )

    args = fake_store.asearch.call_args
    assert args.kwargs["limit"] == 5
