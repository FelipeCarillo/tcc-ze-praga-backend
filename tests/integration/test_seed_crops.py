"""Tests para scripts/seed_crops.py — idempotência.

Mockamos AsyncSession pra não depender de DB real. O foco é confirmar que:
  - quando o select retorna None, um db.add é chamado (cria)
  - quando o select retorna existing, NÃO é chamado db.add (skip)
  - rodar 2x não duplica
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from scripts.seed_crops import CROPS, DISEASES_BY_CROP, seed_crops, seed_diseases


# ── helpers ──────────────────────────────────────────────────────────────────


def _make_session_with_results(results: list) -> AsyncMock:
    """Cria mock session cujo db.execute() retorna ``results`` em sequência.

    Cada item de ``results`` é o objeto retornado por ``.scalar_one_or_none()``
    (None = não existe; algo = já existe).
    """
    session = AsyncMock()
    queue = list(results)

    async def _execute(_stmt):
        sc = MagicMock()
        sc.scalar_one_or_none.return_value = queue.pop(0)
        return sc

    session.execute.side_effect = _execute
    session.commit = AsyncMock()
    session.flush = AsyncMock()
    session.add = MagicMock()
    return session


# ── seed_crops ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_seed_crops_creates_when_missing():
    """Primeira execução — soja não existe, então db.add deve ser chamado."""
    session = _make_session_with_results([None] * len(CROPS))

    slug_to_id = await seed_crops(session)

    assert session.add.call_count == len(CROPS)
    assert session.commit.called
    # slug_to_id é populado mesmo com mock (Crop().id gera uuid no construtor)
    assert "soja" in slug_to_id


@pytest.mark.asyncio
async def test_seed_crops_idempotent_when_existing():
    """Segunda execução — soja já existe, não cria de novo."""
    existing = MagicMock()
    existing.id = "soja-existing-id"
    session = _make_session_with_results([existing] * len(CROPS))

    slug_to_id = await seed_crops(session)

    assert session.add.call_count == 0
    assert slug_to_id["soja"] == "soja-existing-id"


# ── seed_diseases ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_seed_diseases_creates_when_missing():
    n_diseases = sum(len(v) for v in DISEASES_BY_CROP.values())
    session = _make_session_with_results([None] * n_diseases)

    await seed_diseases(session, {"soja": "soja-id"})

    assert session.add.call_count == n_diseases
    assert session.commit.called


@pytest.mark.asyncio
async def test_seed_diseases_idempotent():
    n_diseases = sum(len(v) for v in DISEASES_BY_CROP.values())
    existing = MagicMock()
    session = _make_session_with_results([existing] * n_diseases)

    await seed_diseases(session, {"soja": "soja-id"})

    assert session.add.call_count == 0


@pytest.mark.asyncio
async def test_seed_diseases_skips_when_crop_missing():
    """Sem crop_id mapping, nenhuma disease é criada (sai silently)."""
    session = _make_session_with_results([])  # execute nunca é chamado
    await seed_diseases(session, slug_to_crop_id={})
    assert session.add.call_count == 0


# ── catálogo de soja ─────────────────────────────────────────────────────────


def test_seed_catalog_has_six_soja_diseases():
    """Spec: 6 doenças (5 doenças + saudavel)."""
    assert len(DISEASES_BY_CROP["soja"]) == 6


def test_seed_catalog_disease_slugs_unique_per_crop():
    for crop_slug, diseases in DISEASES_BY_CROP.items():
        slugs = [d["slug"] for d in diseases]
        assert len(slugs) == len(set(slugs)), f"duplicates in {crop_slug}"


def test_seed_catalog_has_saudavel_slug():
    slugs = {d["slug"] for d in DISEASES_BY_CROP["soja"]}
    assert "saudavel" in slugs
