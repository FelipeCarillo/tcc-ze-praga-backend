"""Tests for app/domains/action_plans/repository.py — ActionPlanRepository."""

from unittest.mock import MagicMock

import pytest

from app.domains.action_plans.repository import ActionPlanRepository


def _make_orm_plan(**kwargs):
    p = MagicMock()
    p.disease_id = kwargs.get("disease_id", "ferrugem-asiatica")
    p.level = kwargs.get("level", "essencial")
    p.actions = kwargs.get("actions", ["Ação 1"])
    return p


def _make_orm_source(**kwargs):
    s = MagicMock()
    s.id = kwargs.get("id", "src-1")
    s.disease_id = kwargs.get("disease_id", "ferrugem-asiatica")
    s.name = kwargs.get("name", "EMBRAPA")
    s.detail = kwargs.get("detail", "Fonte")
    s.url = kwargs.get("url", None)
    s.display_order = kwargs.get("display_order", 0)
    return s


# ── find_by_disease ───────────────────────────────────────────────────────────

def _make_execute_result(items: list):
    """Helper: create a MagicMock that mimics an execute result with .scalars().all()."""
    from unittest.mock import MagicMock
    r = MagicMock()
    r.scalars.return_value.all.return_value = items
    return r


async def test_find_by_disease_found(mock_db):
    plans = [_make_orm_plan(level="essencial"), _make_orm_plan(level="campo")]
    sources = [_make_orm_source()]

    # Two separate execute calls → return different MagicMock results
    mock_db.execute.side_effect = [
        _make_execute_result(plans),
        _make_execute_result(sources),
    ]

    repo = ActionPlanRepository(mock_db)
    result = await repo.find_by_disease("ferrugem-asiatica")
    assert result is not None
    assert result.disease_id == "ferrugem-asiatica"
    assert len(result.levels) == 2
    assert len(result.sources) == 1


async def test_find_by_disease_not_found(mock_db):
    mock_db.execute.side_effect = [
        _make_execute_result([]),
        _make_execute_result([]),
    ]
    repo = ActionPlanRepository(mock_db)
    result = await repo.find_by_disease("unknown-disease")
    assert result is None


async def test_find_by_disease_no_sources(mock_db):
    plans = [_make_orm_plan()]
    mock_db.execute.side_effect = [
        _make_execute_result(plans),
        _make_execute_result([]),
    ]
    repo = ActionPlanRepository(mock_db)
    result = await repo.find_by_disease("ferrugem-asiatica")
    assert result is not None
    assert result.sources == []


# ── find_level ────────────────────────────────────────────────────────────────

async def test_find_level_found(mock_db):
    mock_db.execute.return_value.scalar_one_or_none.return_value = _make_orm_plan(level="campo")
    repo = ActionPlanRepository(mock_db)
    result = await repo.find_level("ferrugem-asiatica", "campo")
    assert result is not None
    assert result.level == "campo"


async def test_find_level_not_found(mock_db):
    mock_db.execute.return_value.scalar_one_or_none.return_value = None
    repo = ActionPlanRepository(mock_db)
    result = await repo.find_level("unknown", "essencial")
    assert result is None
