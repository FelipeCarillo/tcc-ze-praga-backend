"""Testes da tool get_disease_info (TCC-041)."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest

from app.domains.chat.tools.get_disease_info import build_get_disease_info_tool
from app.domains.inference.repository import CropDTO, DiseaseDTO


@pytest.fixture(autouse=True)
def _clear_repo_caches():
    """Limpa caches de classe entre testes — eles persistem por processo."""
    from app.domains.inference.repository import (
        CropRepository,
        DiseaseRepository,
    )

    CropRepository.clear_cache()
    DiseaseRepository.clear_cache()
    yield
    CropRepository.clear_cache()
    DiseaseRepository.clear_cache()


def _crop_dto(slug: str = "soja", crop_id: str = "soja-id") -> CropDTO:
    return CropDTO(
        id=crop_id,
        slug=slug,
        name_pt="Soja",
        scientific_name="Glycine max",
        kingdom="plantae",
        is_active=True,
    )


def _disease_dto(slug: str = "ferrugem-asiatica") -> DiseaseDTO:
    return DiseaseDTO(
        id=f"dto-{slug}",
        crop_id="soja-id",
        slug=slug,
        name_pt="Ferrugem Asiática",
        scientific_name="Phakopsora pachyrhizi",
        severity_default="alta",
        description_md="Doença severa.",
        image_url=None,
    )


def _make_session_factory(crop: CropDTO | None, disease: DiseaseDTO | None):
    """Cria session factory que mocka CropRepository + DiseaseRepository calls.

    Estrategia: a sessao em si nao eh usada pelos repos (eles fazem
    ``self._db.execute(...)``) — vamos popular o cache de classe pre-emptive
    pra que os calls reais nao batam o DB.
    """
    from app.domains.inference.repository import (
        CropRepository,
        DiseaseRepository,
    )

    if crop:
        CropRepository._cache_by_slug[crop.slug] = crop
        CropRepository._cache_by_id[crop.id] = crop
    if disease:
        DiseaseRepository._cache_by_crop[disease.crop_id] = [disease]
        DiseaseRepository._cache_by_id[disease.id] = disease

    @asynccontextmanager
    async def factory():
        yield AsyncMock()  # sessao nao usada — cache hit

    return factory


async def test_get_disease_info_returns_known_disease() -> None:
    crop = _crop_dto()
    disease = _disease_dto("mancha-alvo")
    disease = DiseaseDTO(
        id=disease.id,
        crop_id=disease.crop_id,
        slug="mancha-alvo",
        name_pt="Mancha-Alvo",
        scientific_name="Corynespora cassiicola",
        severity_default="media",
        description_md="Mancha em folha.",
        image_url=None,
    )

    factory = _make_session_factory(crop, disease)
    tool = build_get_disease_info_tool(factory)

    raw = await tool.ainvoke(
        {"disease_id": "mancha-alvo", "crop_id": None, "state": {}}
    )
    parsed = json.loads(raw)
    assert parsed["slug"] == "mancha-alvo"
    assert parsed["name_pt"] == "Mancha-Alvo"
    assert parsed["scientific_name"] == "Corynespora cassiicola"


async def test_get_disease_info_handles_unknown_disease() -> None:
    crop = _crop_dto()
    factory = _make_session_factory(crop, None)  # cache vazio de doencas
    # Mas o cache_by_crop fica sem entry — list_by_crop tentaria DB.
    # Vamos popular com lista vazia explicita pra forcar miss.
    from app.domains.inference.repository import DiseaseRepository

    DiseaseRepository._cache_by_crop["soja-id"] = []

    tool = build_get_disease_info_tool(factory)
    raw = await tool.ainvoke(
        {"disease_id": "doenca-fictícia", "crop_id": None, "state": {}}
    )
    parsed = json.loads(raw)
    assert "error" in parsed
    assert "doenca-fictícia" in parsed["error"]


async def test_get_disease_info_uses_state_crop_id() -> None:
    """Quando ``crop_id`` arg eh None, usa ``state.detected_crop_id``."""
    crop = _crop_dto(slug="milho", crop_id="milho-id")
    disease = DiseaseDTO(
        id="dto-ferrugem-milho",
        crop_id="milho-id",
        slug="ferrugem-milho",
        name_pt="Ferrugem do milho",
        scientific_name="Puccinia sorghi",
        severity_default="media",
        description_md=None,
        image_url=None,
    )
    factory = _make_session_factory(crop, disease)
    tool = build_get_disease_info_tool(factory)

    raw = await tool.ainvoke(
        {
            "disease_id": "ferrugem-milho",
            "crop_id": None,
            "state": {"detected_crop_id": "milho"},
        }
    )
    parsed = json.loads(raw)
    assert parsed["slug"] == "ferrugem-milho"
