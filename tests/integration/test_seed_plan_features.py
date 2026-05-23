"""Tests para scripts/seed_plan_features.py — idempotencia (TCC-049).

Mockamos AsyncSession e SubscriptionPlan pra nao depender de DB real. Foco:
  - quando o plano nao existe, status='missing' e nenhum UPDATE roda
  - quando o plano ja tem features iguais, status='skipped' (idempotente)
  - quando features estao desatualizadas, plan.features eh atualizado in-place
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.domains.subscriptions.features import (
    ENTERPRISE_FEATURES,
    FREE_FEATURES,
    PRO_FEATURES,
)
from scripts.seed_plan_features import seed_plan_features


def _make_session_with_results(results: list) -> AsyncMock:
    """Mock session com fila de scalar_one_or_none returns."""
    session = AsyncMock()
    queue = list(results)

    async def _execute(_stmt):
        sc = MagicMock()
        sc.scalar_one_or_none.return_value = queue.pop(0)
        return sc

    session.execute.side_effect = _execute
    session.commit = AsyncMock()
    return session


@pytest.mark.asyncio
async def test_seed_updates_when_features_missing():
    """Plans existem mas features=None -> atualiza."""
    free_plan = MagicMock(features=None)
    pro_plan = MagicMock(features=None)
    enterprise_plan = MagicMock(features=None)
    session = _make_session_with_results([free_plan, pro_plan, enterprise_plan])

    status = await seed_plan_features(session)

    assert status["free"] == "updated"
    assert status["pro"] == "updated"
    assert status["enterprise"] == "updated"
    assert free_plan.features == FREE_FEATURES.model_dump()
    assert pro_plan.features == PRO_FEATURES.model_dump()
    assert enterprise_plan.features == ENTERPRISE_FEATURES.model_dump()
    assert session.commit.called


@pytest.mark.asyncio
async def test_seed_idempotent_when_features_match():
    """Plans com features ja iguais -> status=skipped, sem atualizacao."""
    free_plan = MagicMock(features=FREE_FEATURES.model_dump())
    pro_plan = MagicMock(features=PRO_FEATURES.model_dump())
    enterprise_plan = MagicMock(features=ENTERPRISE_FEATURES.model_dump())
    session = _make_session_with_results([free_plan, pro_plan, enterprise_plan])

    status = await seed_plan_features(session)

    assert status["free"] == "skipped"
    assert status["pro"] == "skipped"
    assert status["enterprise"] == "skipped"


@pytest.mark.asyncio
async def test_seed_marks_missing_when_plan_not_in_db():
    """Quando o plano nao existe no DB, status='missing'."""
    session = _make_session_with_results([None, None, None])

    status = await seed_plan_features(session)

    assert status["free"] == "missing"
    assert status["pro"] == "missing"
    assert status["enterprise"] == "missing"


@pytest.mark.asyncio
async def test_seed_mixed_state():
    """Free=missing, Pro=skipped (iguais), Enterprise=updated (desatualizado)."""
    pro_plan = MagicMock(features=PRO_FEATURES.model_dump())
    enterprise_plan = MagicMock(features={"tier_name": "enterprise", "llm_model": "old"})
    session = _make_session_with_results([None, pro_plan, enterprise_plan])

    status = await seed_plan_features(session)

    assert status["free"] == "missing"
    assert status["pro"] == "skipped"
    assert status["enterprise"] == "updated"
    assert enterprise_plan.features == ENTERPRISE_FEATURES.model_dump()
