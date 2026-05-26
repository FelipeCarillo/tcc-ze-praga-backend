"""Tests for app/domains/diagnoses/service.py — DiagnosisService."""

from unittest.mock import AsyncMock

import pytest

from app.core.exceptions import ForbiddenError, NotFoundError
from app.domains.diagnoses.schemas import CreateDiagnosisRequest, DiagnosisFilters
from app.domains.diagnoses.service import DiagnosisService
from app.shared.enums import SeverityEnum
from tests.conftest import make_diagnosis_dto


@pytest.fixture
def diagnosis_repo():
    repo = AsyncMock()
    repo.create = AsyncMock(return_value=make_diagnosis_dto())
    repo.find_by_id = AsyncMock(return_value=make_diagnosis_dto())
    repo.find_all_by_user = AsyncMock(return_value=([make_diagnosis_dto()], 1))
    repo.delete = AsyncMock(return_value=True)
    repo.delete_all_by_user = AsyncMock(return_value=3)
    return repo


def _create_request() -> CreateDiagnosisRequest:
    return CreateDiagnosisRequest(
        disease_name="Ferrugem",
        disease_id="ferrugem-asiatica",
        confidence=0.9,
        severity=SeverityEnum.ALTA,
        model_used="ensemble",
        top3=[],
    )


# ── create ────────────────────────────────────────────────────────────────────

async def test_create_success(diagnosis_repo):
    svc = DiagnosisService(diagnosis_repo)
    result = await svc.create("user-uuid-1", _create_request(), crop_id="crop-uuid-soja")
    assert result.disease_id == "ferrugem-asiatica"
    diagnosis_repo.create.assert_awaited_once()


# ── get_by_id ─────────────────────────────────────────────────────────────────

async def test_get_by_id_success(diagnosis_repo):
    svc = DiagnosisService(diagnosis_repo)
    result = await svc.get_by_id("diag-uuid-1", "user-uuid-1")
    assert result.id == "diag-uuid-1"


async def test_get_by_id_not_found(diagnosis_repo):
    diagnosis_repo.find_by_id.return_value = None
    svc = DiagnosisService(diagnosis_repo)
    with pytest.raises(NotFoundError, match="Diagnosis"):
        await svc.get_by_id("missing", "user-uuid-1")


async def test_get_by_id_wrong_user(diagnosis_repo):
    """Repo already filters by user_id, but service double-checks ownership."""
    diag = make_diagnosis_dto(user_id="other-user")
    diagnosis_repo.find_by_id.return_value = diag
    svc = DiagnosisService(diagnosis_repo)
    with pytest.raises(ForbiddenError):
        await svc.get_by_id("diag-uuid-1", "user-uuid-1")


# ── list_for_user ─────────────────────────────────────────────────────────────

async def test_list_for_user_returns_paginated(diagnosis_repo):
    svc = DiagnosisService(diagnosis_repo)
    result = await svc.list_for_user("user-uuid-1", DiagnosisFilters())
    assert result.total == 1
    assert result.page == 1
    assert len(result.items) == 1


async def test_list_for_user_empty(diagnosis_repo):
    diagnosis_repo.find_all_by_user.return_value = ([], 0)
    svc = DiagnosisService(diagnosis_repo)
    result = await svc.list_for_user("user-uuid-1", DiagnosisFilters())
    assert result.total == 0
    assert result.items == []


# ── delete ────────────────────────────────────────────────────────────────────

async def test_delete_success(diagnosis_repo):
    svc = DiagnosisService(diagnosis_repo)
    await svc.delete("diag-uuid-1", "user-uuid-1")
    diagnosis_repo.delete.assert_awaited_once()


async def test_delete_not_found(diagnosis_repo):
    diagnosis_repo.delete.return_value = False
    svc = DiagnosisService(diagnosis_repo)
    with pytest.raises(NotFoundError, match="Diagnosis"):
        await svc.delete("missing", "user-uuid-1")


# ── clear_all ─────────────────────────────────────────────────────────────────

async def test_clear_all(diagnosis_repo):
    svc = DiagnosisService(diagnosis_repo)
    count = await svc.clear_all("user-uuid-1")
    assert count == 3


# ── _to_response ──────────────────────────────────────────────────────────────

def test_to_response_mapping():
    dto = make_diagnosis_dto()
    resp = DiagnosisService._to_response(dto)
    assert resp.id == dto.id
    assert resp.confidence == dto.confidence
    assert len(resp.top3) == 1
    # TCC-056: sources default vazio quando DTO nao tem
    assert resp.sources == []


def test_to_response_maps_sources_from_dto():
    """TCC-056: sources JSONB -> DiagnosisSourceSchema."""
    dto = make_diagnosis_dto(
        sources=[
            {
                "type": "web",
                "url": "https://embrapa.br/x",
                "title": "Manejo de ferrugem",
                "snippet": "Aplicar triazol",
                "doi": None,
            },
            {
                "type": "scientific",
                "url": "https://scielo.br/y",
                "title": "Phakopsora study",
                "snippet": "Resistance results",
                "doi": "10.1590/x",
            },
        ]
    )
    resp = DiagnosisService._to_response(dto)
    assert len(resp.sources) == 2
    assert resp.sources[0].type == "web"
    assert resp.sources[1].type == "scientific"
    assert resp.sources[1].doi == "10.1590/x"


def test_to_response_tolerates_malformed_sources():
    """Sources com type invalido caem em fallback 'web'."""
    dto = make_diagnosis_dto(
        sources=[
            {
                "type": "unknown",
                "url": "https://x",
                "title": "T",
            },
            "not-a-dict",  # filtrado
            {"url": "https://y"},  # type ausente -> fallback web
        ]
    )
    resp = DiagnosisService._to_response(dto)
    assert len(resp.sources) == 2
    assert all(s.type == "web" for s in resp.sources)
