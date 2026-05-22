"""Tests para CropRepository + DiseaseRepository — cache em memoria.

Mockamos o AsyncSession (no DB real). Foco:
- Cache hit nao bate execute()
- clear_cache() forca refetch
- DTOs sao frozen (imutaveis)
"""

from dataclasses import FrozenInstanceError
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.domains.inference.repository import (
    CropDTO,
    CropRepository,
    DiseaseDTO,
    DiseaseRepository,
)


# ── helpers ──────────────────────────────────────────────────────────────────


def _orm_crop(**kwargs):
    crop = MagicMock()
    crop.id = kwargs.get("id", "soja-id")
    crop.slug = kwargs.get("slug", "soja")
    crop.name_pt = kwargs.get("name_pt", "Soja")
    crop.scientific_name = kwargs.get("scientific_name", "Glycine max")
    crop.kingdom = kwargs.get("kingdom", "Plantae")
    crop.is_active = kwargs.get("is_active", True)
    return crop


def _orm_disease(**kwargs):
    d = MagicMock()
    d.id = kwargs.get("id", "d-1")
    d.crop_id = kwargs.get("crop_id", "soja-id")
    d.slug = kwargs.get("slug", "ferrugem-asiatica")
    d.name_pt = kwargs.get("name_pt", "Ferrugem")
    d.scientific_name = kwargs.get("scientific_name", "Phakopsora")
    d.severity_default = kwargs.get("severity_default", "alta")
    d.description_md = kwargs.get("description_md", "Desc")
    d.image_url = kwargs.get("image_url", None)
    return d


def _session_returning(scalars_value):
    """Cria mock AsyncSession cujo .execute() retorna result com .scalars() definido."""
    session = AsyncMock()

    async def _execute(_stmt):
        result = MagicMock()
        if isinstance(scalars_value, list):
            result.scalars.return_value.all.return_value = scalars_value
            result.scalar_one_or_none.return_value = (
                scalars_value[0] if scalars_value else None
            )
        else:
            result.scalar_one_or_none.return_value = scalars_value
            result.scalars.return_value.all.return_value = (
                [scalars_value] if scalars_value else []
            )
        return result

    session.execute.side_effect = _execute
    return session


@pytest.fixture(autouse=True)
def _clear_caches():
    CropRepository.clear_cache()
    DiseaseRepository.clear_cache()
    yield
    CropRepository.clear_cache()
    DiseaseRepository.clear_cache()


# ── DTOs ─────────────────────────────────────────────────────────────────────


def test_crop_dto_is_frozen():
    dto = CropDTO(
        id="x",
        slug="y",
        name_pt="Z",
        scientific_name=None,
        kingdom="Plantae",
        is_active=True,
    )
    with pytest.raises(FrozenInstanceError):
        dto.slug = "outro"  # type: ignore[misc]


def test_disease_dto_is_frozen():
    dto = DiseaseDTO(
        id="x",
        crop_id="y",
        slug="z",
        name_pt="N",
        scientific_name=None,
        severity_default="alta",
        description_md=None,
        image_url=None,
    )
    with pytest.raises(FrozenInstanceError):
        dto.slug = "outro"  # type: ignore[misc]


def test_crop_dto_from_orm():
    orm = _orm_crop()
    dto = CropDTO.from_orm(orm)
    assert dto.id == "soja-id"
    assert dto.slug == "soja"


def test_disease_dto_from_orm():
    orm = _orm_disease()
    dto = DiseaseDTO.from_orm(orm)
    assert dto.crop_id == "soja-id"
    assert dto.slug == "ferrugem-asiatica"


# ── CropRepository ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_crop_list_active_caches_results():
    session = _session_returning([_orm_crop()])
    repo = CropRepository(session)
    first = await repo.list_active()
    second = await repo.list_active()
    assert first == second
    assert session.execute.call_count == 1  # cached on second call


@pytest.mark.asyncio
async def test_crop_get_by_slug_uses_cache_after_list():
    session = _session_returning([_orm_crop()])
    repo = CropRepository(session)
    await repo.list_active()
    found = await repo.get_by_slug("soja")
    assert found is not None
    assert session.execute.call_count == 1  # cache hit


@pytest.mark.asyncio
async def test_crop_get_by_slug_returns_none_when_missing():
    session = _session_returning(None)
    repo = CropRepository(session)
    result = await repo.get_by_slug("nao-existe")
    assert result is None


@pytest.mark.asyncio
async def test_crop_get_by_id_returns_none_when_missing():
    session = _session_returning(None)
    repo = CropRepository(session)
    result = await repo.get_by_id("missing-id")
    assert result is None


@pytest.mark.asyncio
async def test_crop_clear_cache_forces_refetch():
    session = _session_returning([_orm_crop()])
    repo = CropRepository(session)
    await repo.list_active()
    CropRepository.clear_cache()
    await repo.list_active()
    assert session.execute.call_count == 2


# ── DiseaseRepository ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_disease_list_by_crop_caches_per_crop():
    session = _session_returning([_orm_disease(id="d-1"), _orm_disease(id="d-2")])
    repo = DiseaseRepository(session)
    first = await repo.list_by_crop("soja-id")
    second = await repo.list_by_crop("soja-id")
    assert first == second
    assert session.execute.call_count == 1


@pytest.mark.asyncio
async def test_disease_list_by_crop_separate_cache_per_crop():
    session = _session_returning([_orm_disease()])
    repo = DiseaseRepository(session)
    await repo.list_by_crop("soja-id")
    await repo.list_by_crop("milho-id")
    assert session.execute.call_count == 2


@pytest.mark.asyncio
async def test_disease_get_by_slug_uses_list_cache():
    session = _session_returning(
        [
            _orm_disease(id="d-1", slug="ferrugem-asiatica"),
            _orm_disease(id="d-2", slug="mancha-alvo"),
        ]
    )
    repo = DiseaseRepository(session)
    found = await repo.get_by_slug("soja-id", "mancha-alvo")
    assert found is not None
    assert found.slug == "mancha-alvo"
    # nova chamada bate o cache
    found2 = await repo.get_by_slug("soja-id", "ferrugem-asiatica")
    assert found2 is not None
    assert session.execute.call_count == 1


@pytest.mark.asyncio
async def test_disease_get_by_slug_returns_none_when_missing():
    session = _session_returning([_orm_disease()])
    repo = DiseaseRepository(session)
    result = await repo.get_by_slug("soja-id", "doenca-fantasia")
    assert result is None


@pytest.mark.asyncio
async def test_disease_get_by_id_returns_none_when_missing():
    session = _session_returning(None)
    repo = DiseaseRepository(session)
    result = await repo.get_by_id("missing-id")
    assert result is None


@pytest.mark.asyncio
async def test_disease_clear_cache_forces_refetch():
    session = _session_returning([_orm_disease()])
    repo = DiseaseRepository(session)
    await repo.list_by_crop("soja-id")
    DiseaseRepository.clear_cache()
    await repo.list_by_crop("soja-id")
    assert session.execute.call_count == 2
