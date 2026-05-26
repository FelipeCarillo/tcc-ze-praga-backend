from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domains.talhoes.dto import TalhaoDTO
from app.domains.talhoes.schemas import CreateTalhaoRequest
from app.models.talhao import Talhao


class TalhaoRepository:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def create(self, user_id: str, data: CreateTalhaoRequest) -> TalhaoDTO:
        talhao = Talhao(
            user_id=user_id,
            nome=data.nome,
            apelido=data.apelido,
            hectares=data.hectares,
            cultura=data.cultura,
            data_semeadura=data.data_semeadura,
        )
        self._db.add(talhao)
        await self._db.commit()
        await self._db.refresh(talhao)
        return self._to_dto(talhao)

    async def find_all_by_user(self, user_id: str) -> list[TalhaoDTO]:
        result = await self._db.execute(
            select(Talhao).where(Talhao.user_id == user_id).order_by(Talhao.created_at.desc())
        )
        return [self._to_dto(t) for t in result.scalars().all()]

    async def find_by_id(self, talhao_id: str, user_id: str) -> TalhaoDTO | None:
        result = await self._db.execute(
            select(Talhao).where(Talhao.id == talhao_id, Talhao.user_id == user_id)
        )
        talhao = result.scalar_one_or_none()
        return self._to_dto(talhao) if talhao else None

    async def delete(self, talhao_id: str, user_id: str) -> bool:
        result = await self._db.execute(
            select(Talhao).where(Talhao.id == talhao_id, Talhao.user_id == user_id)
        )
        talhao = result.scalar_one_or_none()
        if not talhao:
            return False
        await self._db.delete(talhao)
        await self._db.commit()
        return True

    @staticmethod
    def _to_dto(t: Talhao) -> TalhaoDTO:
        return TalhaoDTO(
            id=t.id,
            user_id=t.user_id,
            nome=t.nome,
            apelido=t.apelido,
            hectares=float(t.hectares) if t.hectares is not None else None,
            cultura=t.cultura,
            data_semeadura=t.data_semeadura,
            created_at=t.created_at,
        )
