"""Repositorios para Crop e Disease — leitura cacheada do catalogo.

O catalogo de doencas e crops e' efetivamente read-only em runtime (popula via
seed). Por isso ``list_by_crop`` / ``list_active`` mantem um cache em memoria
indexado por ``crop_id``. Cache pode ser limpo via ``DiseaseRepository.clear_cache()``
caso o seed seja re-executado durante runtime (testes ou re-seed em dev).

Cada repository recebe ``AsyncSession`` por injecao (mesmo padrao dos outros
domains). O cache fica como atributo de classe — partilha entre instancias do
processo, mas e' invalidado por ``clear_cache()``.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.crop import Crop
from app.models.disease import Disease

# ── DTOs ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class CropDTO:
    id: str
    slug: str
    name_pt: str
    scientific_name: str | None
    kingdom: str
    is_active: bool

    @classmethod
    def from_orm(cls, crop: Crop) -> CropDTO:
        return cls(
            id=crop.id,
            slug=crop.slug,
            name_pt=crop.name_pt,
            scientific_name=crop.scientific_name,
            kingdom=crop.kingdom,
            is_active=crop.is_active,
        )


@dataclass(frozen=True, slots=True)
class DiseaseDTO:
    id: str
    crop_id: str
    slug: str
    name_pt: str
    scientific_name: str | None
    severity_default: str
    description_md: str | None
    image_url: str | None

    @classmethod
    def from_orm(cls, disease: Disease) -> DiseaseDTO:
        return cls(
            id=disease.id,
            crop_id=disease.crop_id,
            slug=disease.slug,
            name_pt=disease.name_pt,
            scientific_name=disease.scientific_name,
            severity_default=disease.severity_default,
            description_md=disease.description_md,
            image_url=disease.image_url,
        )


# ── CropRepository ───────────────────────────────────────────────────────────


class CropRepository:
    """Leitura cacheada de crops."""

    _cache_active: list[CropDTO] | None = None
    _cache_by_slug: dict[str, CropDTO] = {}
    _cache_by_id: dict[str, CropDTO] = {}

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def list_active(self) -> list[CropDTO]:
        if type(self)._cache_active is not None:
            return type(self)._cache_active

        result = await self._db.execute(select(Crop).where(Crop.is_active.is_(True)))
        dtos = [CropDTO.from_orm(c) for c in result.scalars().all()]
        type(self)._cache_active = dtos
        for dto in dtos:
            type(self)._cache_by_slug[dto.slug] = dto
            type(self)._cache_by_id[dto.id] = dto
        return dtos

    async def get_by_slug(self, slug: str) -> CropDTO | None:
        if slug in type(self)._cache_by_slug:
            return type(self)._cache_by_slug[slug]

        result = await self._db.execute(select(Crop).where(Crop.slug == slug))
        orm = result.scalar_one_or_none()
        if not orm:
            return None
        dto = CropDTO.from_orm(orm)
        type(self)._cache_by_slug[slug] = dto
        type(self)._cache_by_id[dto.id] = dto
        return dto

    async def get_by_id(self, crop_id: str) -> CropDTO | None:
        if crop_id in type(self)._cache_by_id:
            return type(self)._cache_by_id[crop_id]

        result = await self._db.execute(select(Crop).where(Crop.id == crop_id))
        orm = result.scalar_one_or_none()
        if not orm:
            return None
        dto = CropDTO.from_orm(orm)
        type(self)._cache_by_id[crop_id] = dto
        type(self)._cache_by_slug[dto.slug] = dto
        return dto

    @classmethod
    def clear_cache(cls) -> None:
        cls._cache_active = None
        cls._cache_by_slug = {}
        cls._cache_by_id = {}


# ── DiseaseRepository ────────────────────────────────────────────────────────


class DiseaseRepository:
    """Leitura cacheada de diseases.

    O cache de ``list_by_crop`` e' indexado por ``crop_id`` — invalidacao
    granular por crop fica para o futuro; por ora use ``clear_cache()`` global.
    """

    _cache_by_crop: dict[str, list[DiseaseDTO]] = {}
    _cache_by_id: dict[str, DiseaseDTO] = {}

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def list_by_crop(self, crop_id: str) -> list[DiseaseDTO]:
        if crop_id in type(self)._cache_by_crop:
            return type(self)._cache_by_crop[crop_id]

        result = await self._db.execute(
            select(Disease).where(Disease.crop_id == crop_id)
        )
        dtos = [DiseaseDTO.from_orm(d) for d in result.scalars().all()]
        type(self)._cache_by_crop[crop_id] = dtos
        for dto in dtos:
            type(self)._cache_by_id[dto.id] = dto
        return dtos

    async def get_by_slug(self, crop_id: str, slug: str) -> DiseaseDTO | None:
        # Reutiliza list_by_crop pra cache hit; lookup linear (n=6 hoje).
        for dto in await self.list_by_crop(crop_id):
            if dto.slug == slug:
                return dto
        return None

    async def get_by_id(self, disease_id: str) -> DiseaseDTO | None:
        if disease_id in type(self)._cache_by_id:
            return type(self)._cache_by_id[disease_id]

        result = await self._db.execute(
            select(Disease).where(Disease.id == disease_id)
        )
        orm = result.scalar_one_or_none()
        if not orm:
            return None
        dto = DiseaseDTO.from_orm(orm)
        type(self)._cache_by_id[disease_id] = dto
        return dto

    @classmethod
    def clear_cache(cls) -> None:
        cls._cache_by_crop = {}
        cls._cache_by_id = {}
