"""Integration tests for /api/v1/action-plans router."""

from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.dependencies import (
    get_action_plan_service,
    get_current_user,
    get_plan_features,
)
from app.core.exceptions import NotFoundError
from app.domains.action_plans.schemas import ActionPlanLevelResponse
from app.domains.subscriptions.features import (
    ENTERPRISE_FEATURES,
    FREE_FEATURES,
    PRO_FEATURES,
)
from app.main import app
from app.shared.enums import ActionPlanLevelEnum
from tests.conftest import make_user_dto
from tests.integration.conftest import make_action_plan_dto
from app.domains.action_plans.service import ActionPlanService
from app.domains.action_plans.schemas import ActionPlanResponse, SourceResponse


def _make_action_plan_response() -> ActionPlanResponse:
    """Plano com os TRES niveis — o router e' quem filtra pelo plano."""
    dto = make_action_plan_dto()
    return ActionPlanResponse(
        disease_id=dto.disease_id,
        levels=[
            ActionPlanLevelResponse(
                level=ActionPlanLevelEnum.ESSENCIAL, actions=["Ação 1"]
            ),
            ActionPlanLevelResponse(
                level=ActionPlanLevelEnum.CAMPO, actions=["Ação de campo"]
            ),
            ActionPlanLevelResponse(
                level=ActionPlanLevelEnum.ESPECIALISTA, actions=["Análise técnica"]
            ),
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
    # side_effect (nao return_value): o router muta o objeto ao filtrar niveis,
    # e um unico response compartilhado vazaria estado entre requests do teste.
    svc.get_by_disease = AsyncMock(
        side_effect=lambda *_a, **_k: _make_action_plan_response()
    )
    svc.get_level = AsyncMock(
        return_value=ActionPlanLevelResponse(
            level=ActionPlanLevelEnum.ESSENCIAL, actions=["Ação 1"]
        )
    )
    return svc


@pytest.fixture
async def client_ap(mock_ap_svc):
    """Client Enterprise — enxerga os tres niveis (comportamento pre-gate)."""
    app.dependency_overrides[get_action_plan_service] = lambda: mock_ap_svc
    app.dependency_overrides[get_current_user] = lambda: make_user_dto()
    app.dependency_overrides[get_plan_features] = lambda: ENTERPRISE_FEATURES
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()


def _client_for(mock_ap_svc, features):
    """Helper: client autenticado com um plano especifico."""
    app.dependency_overrides[get_action_plan_service] = lambda: mock_ap_svc
    app.dependency_overrides[get_current_user] = lambda: make_user_dto()
    app.dependency_overrides[get_plan_features] = lambda: features
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


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


# ── Gate por plano (TCC-051) ─────────────────────────────────────────────────
#
# O gate de niveis vivia so' na tool do agente; o REST devolvia tudo pra
# qualquer plano. Como o frontend passou a consumir este endpoint direto, a
# mesma feature ficaria gated no chat e aberta na tela.


async def test_free_recebe_apenas_o_nivel_essencial(mock_ap_svc):
    async with _client_for(mock_ap_svc, FREE_FEATURES) as ac:
        r = await ac.get("/api/v1/action-plans/ferrugem-asiatica")
    app.dependency_overrides.clear()

    assert r.status_code == 200
    body = r.json()
    assert [lvl["level"] for lvl in body["levels"]] == ["essencial"]
    assert body["allowed_levels"] == ["essencial"]


async def test_pro_recebe_essencial_e_campo(mock_ap_svc):
    async with _client_for(mock_ap_svc, PRO_FEATURES) as ac:
        r = await ac.get("/api/v1/action-plans/ferrugem-asiatica")
    app.dependency_overrides.clear()

    assert [lvl["level"] for lvl in r.json()["levels"]] == ["essencial", "campo"]


async def test_enterprise_recebe_os_tres_niveis(mock_ap_svc):
    async with _client_for(mock_ap_svc, ENTERPRISE_FEATURES) as ac:
        r = await ac.get("/api/v1/action-plans/ferrugem-asiatica")
    app.dependency_overrides.clear()

    assert [lvl["level"] for lvl in r.json()["levels"]] == [
        "essencial",
        "campo",
        "especialista",
    ]


async def test_nivel_fora_do_plano_retorna_403(mock_ap_svc):
    """Pedido nominal de um nivel bloqueado: negar, nao rebaixar em silencio."""
    async with _client_for(mock_ap_svc, FREE_FEATURES) as ac:
        r = await ac.get("/api/v1/action-plans/ferrugem-asiatica/especialista")
    app.dependency_overrides.clear()

    assert r.status_code == 403
    assert "especialista" in r.json()["detail"]
    mock_ap_svc.get_level.assert_not_awaited()


async def test_nivel_dentro_do_plano_segue_200(mock_ap_svc):
    async with _client_for(mock_ap_svc, FREE_FEATURES) as ac:
        r = await ac.get("/api/v1/action-plans/ferrugem-asiatica/essencial")
    app.dependency_overrides.clear()

    assert r.status_code == 200
    assert r.json()["level"] == "essencial"
