"""Tests for app/domains/action_plans/service.py — ActionPlanService."""

from unittest.mock import AsyncMock

import pytest

from app.core.exceptions import NotFoundError
from app.domains.action_plans.service import ActionPlanService
from app.shared.enums import ActionPlanLevelEnum
from tests.conftest import make_action_plan_dto
from app.domains.action_plans.dto import ActionPlanLevelDTO


@pytest.fixture
def action_plan_repo():
    repo = AsyncMock()
    repo.find_by_disease = AsyncMock(return_value=make_action_plan_dto())
    repo.find_level = AsyncMock(
        return_value=ActionPlanLevelDTO(
            disease_id="ferrugem-asiatica",
            level="essencial",
            actions=["Ação 1"],
        )
    )
    return repo


# ── get_by_disease ────────────────────────────────────────────────────────────

async def test_get_by_disease_success(action_plan_repo):
    svc = ActionPlanService(action_plan_repo)
    result = await svc.get_by_disease("ferrugem-asiatica")
    assert result.disease_id == "ferrugem-asiatica"
    assert len(result.levels) == 1
    assert len(result.sources) == 1


async def test_get_by_disease_not_found(action_plan_repo):
    action_plan_repo.find_by_disease.return_value = None
    svc = ActionPlanService(action_plan_repo)
    with pytest.raises(NotFoundError, match="ActionPlan"):
        await svc.get_by_disease("unknown-disease")


# ── get_level ─────────────────────────────────────────────────────────────────

async def test_get_level_success(action_plan_repo):
    svc = ActionPlanService(action_plan_repo)
    result = await svc.get_level("ferrugem-asiatica", ActionPlanLevelEnum.ESSENCIAL)
    assert result.level == ActionPlanLevelEnum.ESSENCIAL
    assert "Ação 1" in result.actions


async def test_get_level_not_found(action_plan_repo):
    action_plan_repo.find_level.return_value = None
    svc = ActionPlanService(action_plan_repo)
    with pytest.raises(NotFoundError, match="ActionPlan"):
        await svc.get_level("unknown", ActionPlanLevelEnum.CAMPO)


# ── _to_response ──────────────────────────────────────────────────────────────

def test_to_response_mapping():
    plan = make_action_plan_dto()
    result = ActionPlanService._to_response(plan)
    assert result.disease_id == "ferrugem-asiatica"
    assert result.levels[0].level == ActionPlanLevelEnum.ESSENCIAL
    assert result.sources[0].name == "EMBRAPA"
