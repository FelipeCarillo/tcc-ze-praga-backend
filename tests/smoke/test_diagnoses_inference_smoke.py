"""Smoke tests for diagnoses and inference routes (TCC-070).

Thin layer: verifies routes are wired, require auth, and return valid schema.
Does NOT duplicate the deep assertions from tests/integration/.
"""

from __future__ import annotations

import io
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.dependencies import (
    get_diagnosis_graph_factory,
    get_diagnosis_repository,
    get_diagnosis_service,
    get_inference_service,
    get_store_dep,
)
from app.main import app
from app.shared.pagination import PaginatedResponse
from tests.conftest import make_diagnosis_dto, make_user_dto
from tests.smoke.conftest import bypass_auth_overrides


# -- shared helpers ----------------------------------------------------------


def _make_diagnosis_response():
    from app.domains.diagnoses.schemas import DiagnosisResponse, Top3PredictionSchema

    d = make_diagnosis_dto()
    return DiagnosisResponse(
        id=d.id,
        disease_name=d.disease_name,
        disease_id=d.disease_id,
        scientific_name=d.scientific_name,
        confidence=d.confidence,
        severity=d.severity,
        description=d.description,
        model_used=d.model_used,
        image_url=d.image_url,
        image_name=d.image_name,
        created_at=d.created_at,
        top3=[
            Top3PredictionSchema(
                rank=t.rank,
                disease_name=t.disease_name,
                disease_id=t.disease_id,
                scientific_name=t.scientific_name,
                confidence=t.confidence,
                severity=t.severity,
            )
            for t in d.top3
        ],
    )


# -- /diagnoses/analyze fixture -----------------------------------------------


@pytest.fixture
async def client_analyze():
    """Client for POST /api/v1/diagnoses/analyze.

    bypass_auth_overrides() unblocks require_quota_dual by overriding
    get_current_user / get_current_user_or_api_key / get_usage_service.
    """
    bypass_auth_overrides()

    graph = MagicMock()
    graph.ainvoke = AsyncMock(return_value={"persisted_ids": ["diag-0"]})
    factory_fn = MagicMock(return_value=graph)

    mock_diag_repo = AsyncMock()
    mock_diag_repo.find_by_id = AsyncMock(
        side_effect=lambda diag_id, _user_id: make_diagnosis_dto(id=diag_id)
    )

    app.dependency_overrides[get_diagnosis_graph_factory] = lambda: factory_fn
    app.dependency_overrides[get_diagnosis_repository] = lambda: mock_diag_repo

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()


# -- /diagnoses CRUD fixture --------------------------------------------------


@pytest.fixture
async def client_diag():
    """Client for the standard CRUD routes (POST/GET/DELETE /diagnoses)."""
    bypass_auth_overrides()

    diag_response = _make_diagnosis_response()

    mock_diag_svc = AsyncMock()
    mock_diag_svc.create = AsyncMock(return_value=diag_response)
    mock_diag_svc.get_by_id = AsyncMock(return_value=diag_response)
    mock_diag_svc.list_for_user = AsyncMock(
        return_value=PaginatedResponse(
            items=[diag_response], total=1, page=1, limit=20
        )
    )
    mock_diag_svc.delete = AsyncMock()
    mock_diag_svc.clear_all = AsyncMock(return_value=2)

    mock_inference_svc = SimpleNamespace(
        disease_catalog=[SimpleNamespace(crop_id="crop-uuid-soja")]
    )

    app.dependency_overrides[get_diagnosis_service] = lambda: mock_diag_svc
    app.dependency_overrides[get_inference_service] = lambda: mock_inference_svc

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac, mock_diag_svc
    app.dependency_overrides.clear()


# -- /diagnoses/semantic fixture ----------------------------------------------


@pytest.fixture
async def client_semantic():
    """Client for GET /api/v1/diagnoses/semantic."""
    bypass_auth_overrides()

    mock_store = MagicMock()
    mock_store.asearch = AsyncMock(return_value=[])

    app.dependency_overrides[get_store_dep] = lambda: mock_store

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()


# -- /inference fixture -------------------------------------------------------


@pytest.fixture
async def client_inference():
    """Client for POST /api/v1/inference."""
    bypass_auth_overrides()

    from app.domains.inference.schemas import InferenceResult
    from app.domains.diagnoses.schemas import Top3PredictionSchema
    from app.shared.enums import SeverityEnum

    fake_result = InferenceResult(
        disease_id="ferrugem-asiatica",
        disease_name="Ferrugem Asiatica",
        scientific_name="Phakopsora pachyrhizi",
        severity=SeverityEnum.ALTA,
        description="Doenca fungica severa.",
        confidence=0.94,
        model_id="ensemble",
        image_name="folha.jpg",
        top3=[
            Top3PredictionSchema(
                rank=1,
                disease_name="Ferrugem Asiatica",
                disease_id="ferrugem-asiatica",
                scientific_name="Phakopsora pachyrhizi",
                confidence=0.94,
                severity=SeverityEnum.ALTA,
            )
        ],
    )

    mock_inference_svc = MagicMock()
    mock_inference_svc.predict = MagicMock(return_value=fake_result)
    mock_inference_svc.disease_catalog = [SimpleNamespace(crop_id="crop-uuid-soja")]

    diag_response = _make_diagnosis_response()
    mock_diag_svc = AsyncMock()
    mock_diag_svc.create = AsyncMock(return_value=diag_response)

    app.dependency_overrides[get_inference_service] = lambda: mock_inference_svc
    app.dependency_overrides[get_diagnosis_service] = lambda: mock_diag_svc

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()


# -- 401 without auth ---------------------------------------------------------


async def test_diagnoses_list_without_auth_returns_401(smoke_client):
    """GET /diagnoses requires auth -- smoke_client has no overrides."""
    r = await smoke_client.get("/api/v1/diagnoses")
    assert r.status_code == 401


# -- POST /diagnoses/analyze --------------------------------------------------


async def test_analyze_happy_path(client_analyze):
    files = {"images": ("leaf.jpg", io.BytesIO(b"\x89PNG"), "image/jpeg")}
    data = {"crop_id": "soja", "model": "ensemble"}
    r = await client_analyze.post("/api/v1/diagnoses/analyze", data=data, files=files)
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body, list)
    assert body[0]["id"] == "diag-0"


# -- POST /diagnoses ----------------------------------------------------------


async def test_create_diagnosis_201(client_diag):
    ac, _ = client_diag
    payload = {
        "disease_name": "Ferrugem Asiatica",
        "disease_id": "ferrugem-asiatica",
        "confidence": 0.94,
        "severity": "alta",
        "model_used": "ensemble",
        "top3": [],
    }
    r = await ac.post("/api/v1/diagnoses", json=payload)
    assert r.status_code == 201
    assert r.json()["disease_id"] == "ferrugem-asiatica"


# -- GET /diagnoses -----------------------------------------------------------


async def test_list_diagnoses_200(client_diag):
    ac, _ = client_diag
    r = await ac.get("/api/v1/diagnoses")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 1
    assert len(body["items"]) == 1


# -- GET /diagnoses/{id} ------------------------------------------------------


async def test_get_diagnosis_by_id(client_diag):
    ac, _ = client_diag
    r = await ac.get("/api/v1/diagnoses/diag-uuid-1")
    assert r.status_code == 200
    assert r.json()["id"] == "diag-uuid-1"


# -- DELETE /diagnoses/{id} ---------------------------------------------------


async def test_delete_diagnosis_204(client_diag):
    ac, _ = client_diag
    r = await ac.delete("/api/v1/diagnoses/diag-uuid-1")
    assert r.status_code == 204


# -- DELETE /diagnoses?confirm=true -------------------------------------------


async def test_clear_all_with_confirm(client_diag):
    ac, _ = client_diag
    r = await ac.delete("/api/v1/diagnoses?confirm=true")
    assert r.status_code == 200
    assert "deleted" in r.json()


# -- GET /diagnoses/semantic --------------------------------------------------


async def test_semantic_search_returns_list(client_semantic):
    r = await client_semantic.get("/api/v1/diagnoses/semantic?q=ferrugem")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


# -- POST /inference ----------------------------------------------------------


async def test_run_inference_201(client_inference):
    files = {"image": ("folha.jpg", io.BytesIO(b"\x89PNG"), "image/jpeg")}
    data = {"model": "ensemble"}
    r = await client_inference.post("/api/v1/inference", data=data, files=files)
    assert r.status_code == 201
    assert r.json()["disease_id"] == "ferrugem-asiatica"
