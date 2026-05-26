"""Integration tests for /api/v1/diagnoses router."""

from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.dependencies import (
    get_current_user,
    get_diagnosis_service,
    get_inference_service,
    get_usage_service,
    require_quota,
)
from app.core.exceptions import ForbiddenError, NotFoundError, QuotaExceededError
from app.main import app
from app.shared.enums import FeatureTypeEnum
from app.shared.pagination import PaginatedResponse
from tests.conftest import make_user_dto
from tests.integration.conftest import make_diagnosis_response, make_usage_summary

VALID_BODY = {
    "disease_name": "Ferrugem Asiática",
    "disease_id": "ferrugem-asiatica",
    "confidence": 0.94,
    "severity": "alta",
    "model_used": "ensemble",
    "top3": [],
}


@pytest.fixture
def mock_diag_svc():
    svc = AsyncMock()
    svc.create = AsyncMock(return_value=make_diagnosis_response())
    svc.get_by_id = AsyncMock(return_value=make_diagnosis_response())
    svc.list_for_user = AsyncMock(
        return_value=PaginatedResponse(
            items=[make_diagnosis_response()], total=1, page=1, limit=20
        )
    )
    svc.delete = AsyncMock()
    svc.clear_all = AsyncMock(return_value=3)
    return svc


@pytest.fixture
def mock_usage_svc():
    svc = AsyncMock()
    svc.check_quota = AsyncMock()
    svc.record_usage = AsyncMock()
    svc.get_summary = AsyncMock(return_value=make_usage_summary())
    svc.get_history = AsyncMock(return_value=[])
    return svc


@pytest.fixture
def mock_inference_svc():
    """Stub do InferenceService — `create_diagnosis` le `disease_catalog[0].crop_id`.

    Sem o override, `get_inference_service` real chama `crop_repo.get_by_slug("soja")`
    e dispara `SELECT FROM crops` no Postgres de teste, que nao tem a tabela.
    """
    from types import SimpleNamespace

    svc = SimpleNamespace(
        disease_catalog=[SimpleNamespace(crop_id="crop-uuid-soja")],
    )
    return svc


@pytest.fixture
async def client_diag(mock_diag_svc, mock_usage_svc, mock_inference_svc):
    # Override require_quota to bypass quota check
    app.dependency_overrides[require_quota(FeatureTypeEnum.INFERENCE)] = lambda: make_user_dto()
    app.dependency_overrides[get_diagnosis_service] = lambda: mock_diag_svc
    app.dependency_overrides[get_usage_service] = lambda: mock_usage_svc
    app.dependency_overrides[get_inference_service] = lambda: mock_inference_svc
    app.dependency_overrides[get_current_user] = lambda: make_user_dto()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()


# ── POST /diagnoses ───────────────────────────────────────────────────────────

async def test_create_diagnosis_201(client_diag):
    r = await client_diag.post("/api/v1/diagnoses", json=VALID_BODY)
    assert r.status_code == 201
    assert r.json()["disease_id"] == "ferrugem-asiatica"


async def test_create_diagnosis_invalid_body(client_diag):
    r = await client_diag.post("/api/v1/diagnoses", json={"disease_name": "X"})
    assert r.status_code == 422


# ── GET /diagnoses ────────────────────────────────────────────────────────────

async def test_list_diagnoses_200(client_diag):
    r = await client_diag.get("/api/v1/diagnoses")
    assert r.status_code == 200
    data = r.json()
    assert data["total"] == 1
    assert len(data["items"]) == 1


async def test_list_diagnoses_with_filters(client_diag):
    r = await client_diag.get("/api/v1/diagnoses?severity=alta&search=ferrugem&page=1&limit=5")
    assert r.status_code == 200


# ── GET /diagnoses/{id} ───────────────────────────────────────────────────────

async def test_get_diagnosis_200(client_diag):
    r = await client_diag.get("/api/v1/diagnoses/diag-uuid-1")
    assert r.status_code == 200
    assert r.json()["id"] == "diag-uuid-1"


async def test_get_diagnosis_not_found(client_diag, mock_diag_svc):
    mock_diag_svc.get_by_id.side_effect = NotFoundError("Diagnosis", "missing")
    r = await client_diag.get("/api/v1/diagnoses/missing")
    assert r.status_code == 404


async def test_get_diagnosis_forbidden(client_diag, mock_diag_svc):
    mock_diag_svc.get_by_id.side_effect = ForbiddenError()
    r = await client_diag.get("/api/v1/diagnoses/other-user-diag")
    assert r.status_code == 403


# ── DELETE /diagnoses/{id} ────────────────────────────────────────────────────

async def test_delete_diagnosis_204(client_diag):
    r = await client_diag.delete("/api/v1/diagnoses/diag-uuid-1")
    assert r.status_code == 204


async def test_delete_diagnosis_not_found(client_diag, mock_diag_svc):
    mock_diag_svc.delete.side_effect = NotFoundError("Diagnosis", "missing")
    r = await client_diag.delete("/api/v1/diagnoses/missing")
    assert r.status_code == 404


# ── DELETE /diagnoses ─────────────────────────────────────────────────────────

async def test_clear_all_without_confirm(client_diag):
    r = await client_diag.delete("/api/v1/diagnoses")
    assert r.status_code == 200
    assert "confirm=true" in r.json()["detail"].lower()


async def test_clear_all_with_confirm(client_diag):
    r = await client_diag.delete("/api/v1/diagnoses?confirm=true")
    assert r.status_code == 200
    assert r.json()["deleted"] == 3
