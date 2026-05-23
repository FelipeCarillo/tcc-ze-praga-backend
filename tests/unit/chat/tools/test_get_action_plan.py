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


# ── TCC-051: plan-features gating ───────────────────────────────────────────


def _free_features():
    from app.domains.subscriptions.features import FREE_FEATURES

    return FREE_FEATURES


def _pro_features():
    from app.domains.subscriptions.features import PRO_FEATURES

    return PRO_FEATURES


def _enterprise_features():
    from app.domains.subscriptions.features import ENTERPRISE_FEATURES

    return ENTERPRISE_FEATURES


async def test_free_plan_only_returns_essencial_level() -> None:
    svc = AsyncMock()
    svc.get_by_disease.return_value = _plan()
    tool = build_get_action_plan_tool(svc)

    text = await tool.ainvoke(
        {
            "disease_id": "ferrugem-asiatica",
            "state": {"plan_features": _free_features()},
        }
    )
    assert "ESSENCIAL" in text
    assert "CAMPO" not in text  # Free nao tem campo


async def test_pro_plan_returns_essencial_and_campo() -> None:
    svc = AsyncMock()
    svc.get_by_disease.return_value = _plan()
    tool = build_get_action_plan_tool(svc)

    text = await tool.ainvoke(
        {
            "disease_id": "ferrugem-asiatica",
            "state": {"plan_features": _pro_features()},
        }
    )
    assert "ESSENCIAL" in text
    assert "CAMPO" in text


async def test_enterprise_plan_returns_all_levels() -> None:
    from app.domains.action_plans.schemas import (
        ActionPlanLevelResponse,
        ActionPlanResponse,
    )
    from app.shared.enums import ActionPlanLevelEnum

    plan_with_all_levels = ActionPlanResponse(
        disease_id="ferrugem-asiatica",
        levels=[
            ActionPlanLevelResponse(
                level=ActionPlanLevelEnum.ESSENCIAL, actions=["A"]
            ),
            ActionPlanLevelResponse(level=ActionPlanLevelEnum.CAMPO, actions=["B"]),
            ActionPlanLevelResponse(
                level=ActionPlanLevelEnum.ESPECIALISTA, actions=["C"]
            ),
        ],
        sources=[],
    )
    svc = AsyncMock()
    svc.get_by_disease.return_value = plan_with_all_levels
    tool = build_get_action_plan_tool(svc)

    text = await tool.ainvoke(
        {
            "disease_id": "ferrugem-asiatica",
            "state": {"plan_features": _enterprise_features()},
        }
    )
    assert "ESSENCIAL" in text
    assert "CAMPO" in text
    assert "ESPECIALISTA" in text


async def test_level_arg_filters_to_single_level() -> None:
    svc = AsyncMock()
    svc.get_by_disease.return_value = _plan()
    tool = build_get_action_plan_tool(svc)

    text = await tool.ainvoke(
        {
            "disease_id": "ferrugem-asiatica",
            "level": "campo",
            "state": {"plan_features": _pro_features()},
        }
    )
    assert "CAMPO" in text
    assert "ESSENCIAL" not in text


async def test_level_arg_not_in_plan_shows_warning() -> None:
    """Free pede 'campo' — nao permitido, mostra warning."""
    svc = AsyncMock()
    svc.get_by_disease.return_value = _plan()
    tool = build_get_action_plan_tool(svc)

    text = await tool.ainvoke(
        {
            "disease_id": "ferrugem-asiatica",
            "level": "campo",
            "state": {"plan_features": _free_features()},
        }
    )
    assert "nao disponivel" in text or "não disponível" in text or "essencial" in text.lower()


async def test_no_plan_features_means_all_levels_returned() -> None:
    """Back-compat: state sem plan_features -> tool nao filtra."""
    svc = AsyncMock()
    svc.get_by_disease.return_value = _plan()
    tool = build_get_action_plan_tool(svc)

    text = await tool.ainvoke({"disease_id": "ferrugem-asiatica", "state": {}})
    assert "ESSENCIAL" in text
    assert "CAMPO" in text
