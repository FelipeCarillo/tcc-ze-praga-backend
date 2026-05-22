"""Testes da tool get_action_plan (TCC-041)."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app.domains.action_plans.schemas import (
    ActionPlanLevelResponse,
    ActionPlanResponse,
    SourceResponse,
)
from app.domains.chat.tools.get_action_plan import build_get_action_plan_tool
from app.shared.enums import ActionPlanLevelEnum


def _plan(disease_id: str = "ferrugem-asiatica") -> ActionPlanResponse:
    return ActionPlanResponse(
        disease_id=disease_id,
        levels=[
            ActionPlanLevelResponse(
                level=ActionPlanLevelEnum.ESSENCIAL,
                actions=["Aplicar fungicida"],
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


async def test_get_action_plan_returns_text() -> None:
    svc = AsyncMock()
    svc.get_by_disease.return_value = _plan()
    tool = build_get_action_plan_tool(svc)

    text = await tool.ainvoke(
        {"disease_id": "ferrugem-asiatica", "state": {}}
    )
    assert "Plano de acao para ferrugem-asiatica" in text
    assert "ESSENCIAL" in text
    assert "CAMPO" in text
    assert "Aplicar fungicida" in text
    assert "EMBRAPA" in text
    svc.get_by_disease.assert_awaited_once_with("ferrugem-asiatica")


async def test_get_action_plan_filters_by_preferred_level() -> None:
    svc = AsyncMock()
    svc.get_by_disease.return_value = _plan()
    tool = build_get_action_plan_tool(svc)

    text = await tool.ainvoke(
        {
            "disease_id": "ferrugem-asiatica",
            "state": {"preferred_action_level": "essencial"},
        }
    )
    assert "ESSENCIAL" in text
    assert "CAMPO" not in text


async def test_get_action_plan_handles_not_found() -> None:
    svc = AsyncMock()
    svc.get_by_disease.side_effect = ValueError("nope")
    tool = build_get_action_plan_tool(svc)

    text = await tool.ainvoke({"disease_id": "unknown", "state": {}})
    assert "indisponivel" in text.lower()
