"""Integration tests for /api/v1/action-plans router."""

from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.dependencies import get_action_plan_service, get_current_user
from app.core.exceptions import NotFoundError
from app.domains.action_plans.schemas import ActionPlanLevelResponse
from app.main import app
from app.shared.enums import ActionPlanLevelEnum
from tests.conftest import make_user_dto
from tests.integration.conftest import make_action_plan_dto
from app.domains.action_plans.service import ActionPlanService
from app.domains.action_plans.schemas import ActionPlanResponse, SourceResponse


def _make_action_plan_response() -> ActionPlanResponse:
    dto = make_action_plan_dto()
    return ActionPlanResponse(
        disease_id=dto.disease_id,
        levels=[
            ActionPlanLevelResponse(level=ActionPlanLevelEnum.ESSENCIAL, actions=["Ação 1"])
        ],
        sources=[
            SourceResponse(
                id="src-1",
                name="EMBRAPA",
                detail="Fonte",
                url=None,
                display_order=0,
            )
        ],
    )


@pytest.fixture
def mock_ap_svc():
    svc = AsyncMock()
    svc.get_by_disease = AsyncMock(return_value=_make_action_plan_response())
    svc.get_level = AsyncMock(
        return_value=ActionPlanLevelResponse(
            level=ActionPlanLevelEnum.ESSENCIAL, actions=["Ação 1"]
        )
    )
    return svc


@pytest.fixture
async def client_ap(mock_ap_svc):
    app.dependency_overrides[get_action_plan_service] = lambda: mock_ap_svc
    app.dependency_overrides[get_current_user] = lambda: make_user_dto()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()


async def test_get_action_plan_200(client_ap):
    r = await client_ap.get("/api/v1/action-plans/ferrugem-asiatica")
    assert r.status_code == 200
    assert r.json()["disease_id"] == "ferrugem-asiatica"


async def test_get_action_plan_not_found(client_ap, mock_ap_svc):
    mock_ap_svc.get_by_disease.side_effect = NotFoundError("ActionPlan", "unknown")
    r = await client_ap.get("/api/v1/action-plans/unknown")
    assert r.status_code == 404


async def test_get_action_plan_level_200(client_ap):
    r = await client_ap.get("/api/v1/action-plans/ferrugem-asiatica/essencial")
    assert r.status_code == 200
    assert r.json()["level"] == "essencial"


async def test_get_action_plan_level_not_found(client_ap, mock_ap_svc):
    mock_ap_svc.get_level.side_effect = NotFoundError("ActionPlan", "unknown/essencial")
    r = await client_ap.get("/api/v1/action-plans/unknown/essencial")
    assert r.status_code == 404
