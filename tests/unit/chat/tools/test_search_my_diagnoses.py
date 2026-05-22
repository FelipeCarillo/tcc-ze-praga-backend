"""Testes do placeholder search_my_diagnoses (TCC-041)."""

from __future__ import annotations

import json

from app.domains.chat.tools.search_my_diagnoses import (
    build_search_my_diagnoses_tool,
)


async def test_search_my_diagnoses_returns_empty_list() -> None:
    tool = build_search_my_diagnoses_tool()
    raw = await tool.ainvoke(
        {"query": "ferrugem", "limit": 5, "state": {}}
    )
    parsed = json.loads(raw)
    assert parsed == []


async def test_search_my_diagnoses_default_limit() -> None:
    tool = build_search_my_diagnoses_tool()
    # Default limit kicks in when not passed
    raw = await tool.ainvoke({"query": "qualquer", "state": {}})
    assert raw == "[]"
