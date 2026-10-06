from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domains.fazendas.dto import FazendaDTO
from app.domains.fazendas.schemas import CreateFazendaRequest, UpdateFazendaRequest
from app.models.fazenda import Fazenda

DEFAULT_NOME = "Minha fazenda"


class FazendaRepository:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def create(self, user_id: str, data: CreateFazendaRequest) -> FazendaDTO:
        fazenda = Fazenda(user_id=user_id, **data.model_dump())
        self._db.add(fazenda)
        await self._db.commit()
        await self._db.refresh(fazenda)
        return self._to_dto(fazenda)

    async def find_all_by_user(self, user_id: str) -> list[FazendaDTO]:
        result = await self._db.execute(
            select(Fazenda).where(Fazenda.user_id == user_id).order_by(Fazenda.created_at)
        )
        return [self._to_dto(f) for f in result.scalars().all()]

    async def find_by_id(self, fazenda_id: str, user_id: str) -> FazendaDTO | None:
        fazenda = await self._get(fazenda_id, user_id)
        return self._to_dto(fazenda) if fazenda else None

    async def get_or_create_default(self, user_id: str) -> FazendaDTO:
        """A fazenda mais antiga do usuário; sem nenhuma, cria "Minha fazenda".

        Mantém o ``POST /talhoes`` sem ``fazenda_id`` funcionando (clientes
        antigos e o agente) sem deixar talhão órfão.
        """
        existentes = await self.find_all_by_user(user_id)
        if existentes:
            return existentes[0]
        return await self.create(user_id, CreateFazendaRequest(nome=DEFAULT_NOME))

    async def update(
        self, fazenda_id: str, user_id: str, data: UpdateFazendaRequest
    ) -> FazendaDTO | None:
        fazenda = await self._get(fazenda_id, user_id)
        if not fazenda:
            return None
        for campo, valor in data.model_dump(exclude_unset=True).items():
            setattr(fazenda, campo, valor)
        await self._db.commit()
        await self._db.refresh(fazenda)
        return self._to_dto(fazenda)

    async def delete(self, fazenda_id: str, user_id: str) -> bool:
        fazenda = await self._get(fazenda_id, user_id)
        if not fazenda:
            return False
        await self._db.delete(fazenda)
        await self._db.commit()
        return True

    async def _get(self, fazenda_id: str, user_id: str) -> Fazenda | None:
        result = await self._db.execute(
            select(Fazenda).where(Fazenda.id == fazenda_id, Fazenda.user_id == user_id)
        )
        return result.scalar_one_or_none()

    @staticmethod
    def _to_dto(f: Fazenda) -> FazendaDTO:
        return FazendaDTO(
            id=f.id,
            user_id=f.user_id,
            nome=f.nome,
            municipio=f.municipio,
            uf=f.uf,
            hectares=float(f.hectares) if f.hectares is not None else None,
            agronomo_nome=f.agronomo_nome,
            agronomo_crea=f.agronomo_crea,
            created_at=f.created_at,
        )
