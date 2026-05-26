"""Tests for app/domains/diagnoses/repository.py — DiagnosisRepository."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.domains.diagnoses.repository import DiagnosisRepository
from app.domains.diagnoses.schemas import CreateDiagnosisRequest, DiagnosisFilters
from app.shared.enums import SeverityEnum


def _make_orm_top3(rank=1, **kwargs):
    t = MagicMock()
    t.rank = rank
    t.disease_name = kwargs.get("disease_name", "Ferrugem")
    t.disease_id = kwargs.get("disease_id", "ferrugem-asiatica")
    t.scientific_name = kwargs.get("scientific_name", None)
    t.confidence = 0.9
    t.severity = kwargs.get("severity", "alta")
    return t


def _make_orm_diagnosis(**kwargs):
    d = MagicMock()
    d.id = kwargs.get("id", "diag-1")
    d.user_id = kwargs.get("user_id", "user-1")
    d.disease_name = kwargs.get("disease_name", "Ferrugem Asiática")
    d.disease_id = kwargs.get("disease_id", "ferrugem-asiatica")
    d.scientific_name = kwargs.get("scientific_name", None)
    d.confidence = 0.942
    d.severity = kwargs.get("severity", "alta")
    d.description = kwargs.get("description", "Desc")
    d.model_used = kwargs.get("model_used", "ensemble")
    d.image_url = kwargs.get("image_url", None)
    d.image_name = kwargs.get("image_name", None)
    d.created_at = kwargs.get("created_at", datetime(2026, 4, 1, tzinfo=UTC))
    d.top3 = kwargs.get("top3", [_make_orm_top3()])
    return d


def _make_create_request(**kwargs) -> CreateDiagnosisRequest:
    return CreateDiagnosisRequest(
        disease_name=kwargs.get("disease_name", "Ferrugem"),
        disease_id=kwargs.get("disease_id", "ferrugem-asiatica"),
        confidence=kwargs.get("confidence", 0.9),
        severity=SeverityEnum.ALTA,
        model_used=kwargs.get("model_used", "ensemble"),
        top3=[],
    )


# ── create ────────────────────────────────────────────────────────────────────

async def test_create_diagnosis(mock_db):
    orm_diag = _make_orm_diagnosis()
    mock_db.flush = AsyncMock()

    # find_by_id is called internally after create; simulate via execute
    mock_db.execute.return_value.unique.return_value.scalar_one_or_none.return_value = orm_diag

    repo = DiagnosisRepository(mock_db)
    result = await repo.create("user-1", _make_create_request(), crop_id="crop-uuid-soja")

    mock_db.add.assert_called()
    mock_db.flush.assert_awaited()
    mock_db.commit.assert_awaited()
    assert result is not None
    assert result.disease_name == "Ferrugem Asiática"


async def test_create_diagnosis_with_top3(mock_db):
    from app.domains.diagnoses.schemas import Top3PredictionSchema

    orm_diag = _make_orm_diagnosis()
    mock_db.execute.return_value.unique.return_value.scalar_one_or_none.return_value = orm_diag

    request = CreateDiagnosisRequest(
        disease_name="Ferrugem",
        disease_id="ferrugem-asiatica",
        confidence=0.9,
        severity=SeverityEnum.ALTA,
        model_used="ensemble",
        top3=[
            Top3PredictionSchema(
                rank=1,
                disease_name="Ferrugem",
                disease_id="ferrugem-asiatica",
                confidence=0.9,
            )
        ],
    )
    repo = DiagnosisRepository(mock_db)
    result = await repo.create("user-1", request, crop_id="crop-uuid-soja")
    assert result is not None


# ── find_by_id ────────────────────────────────────────────────────────────────

async def test_find_by_id_found(mock_db):
    orm_diag = _make_orm_diagnosis()
    mock_db.execute.return_value.unique.return_value.scalar_one_or_none.return_value = orm_diag
    repo = DiagnosisRepository(mock_db)
    result = await repo.find_by_id("diag-1", "user-1")
    assert result is not None
    assert result.id == "diag-1"


async def test_find_by_id_not_found(mock_db):
    mock_db.execute.return_value.unique.return_value.scalar_one_or_none.return_value = None
    repo = DiagnosisRepository(mock_db)
    result = await repo.find_by_id("missing", "user-1")
    assert result is None


# ── find_all_by_user ──────────────────────────────────────────────────────────

async def test_find_all_no_filters(mock_db):
    diags = [_make_orm_diagnosis()]
    mock_db.execute.return_value.scalar_one.return_value = 1
    mock_db.execute.return_value.unique.return_value.scalars.return_value.all.return_value = diags
    repo = DiagnosisRepository(mock_db)
    items, total = await repo.find_all_by_user("user-1", DiagnosisFilters())
    assert total == 1
    assert len(items) == 1


async def test_find_all_with_severity_filter(mock_db):
    mock_db.execute.return_value.scalar_one.return_value = 0
    mock_db.execute.return_value.unique.return_value.scalars.return_value.all.return_value = []
    repo = DiagnosisRepository(mock_db)
    items, total = await repo.find_all_by_user(
        "user-1", DiagnosisFilters(severity=SeverityEnum.ALTA)
    )
    assert total == 0


async def test_find_all_with_search_filter(mock_db):
    mock_db.execute.return_value.scalar_one.return_value = 0
    mock_db.execute.return_value.unique.return_value.scalars.return_value.all.return_value = []
    repo = DiagnosisRepository(mock_db)
    items, total = await repo.find_all_by_user(
        "user-1", DiagnosisFilters(search="ferrugem")
    )
    assert total == 0


# ── delete ────────────────────────────────────────────────────────────────────

async def test_delete_found(mock_db):
    orm_diag = _make_orm_diagnosis()
    mock_db.execute.return_value.scalar_one_or_none.return_value = orm_diag
    repo = DiagnosisRepository(mock_db)
    result = await repo.delete("diag-1", "user-1")
    assert result is True
    mock_db.delete.assert_awaited_once()
    mock_db.commit.assert_awaited()


async def test_delete_not_found(mock_db):
    mock_db.execute.return_value.scalar_one_or_none.return_value = None
    repo = DiagnosisRepository(mock_db)
    result = await repo.delete("missing", "user-1")
    assert result is False


# ── delete_all_by_user ────────────────────────────────────────────────────────

async def test_delete_all(mock_db):
    diags = [_make_orm_diagnosis(id=f"d-{i}") for i in range(3)]
    mock_db.execute.return_value.scalars.return_value.all.return_value = diags
    repo = DiagnosisRepository(mock_db)
    count = await repo.delete_all_by_user("user-1")
    assert count == 3
    assert mock_db.delete.await_count == 3


async def test_delete_all_empty(mock_db):
    mock_db.execute.return_value.scalars.return_value.all.return_value = []
    repo = DiagnosisRepository(mock_db)
    count = await repo.delete_all_by_user("user-1")
    assert count == 0


# ── _to_dto ───────────────────────────────────────────────────────────────────

def test_to_dto_maps_correctly():
    orm_diag = _make_orm_diagnosis(top3=[])
    dto = DiagnosisRepository._to_dto(orm_diag)
    assert dto.confidence == float(orm_diag.confidence)
    assert dto.top3 == []


def test_to_dto_with_top3():
    top3 = [_make_orm_top3(rank=1), _make_orm_top3(rank=2)]
    orm_diag = _make_orm_diagnosis(top3=top3)
    dto = DiagnosisRepository._to_dto(orm_diag)
    assert len(dto.top3) == 2
    assert dto.top3[0].rank == 1
