from typing import TYPE_CHECKING

from app.core.exceptions import NotFoundError
from app.domains.talhoes.dto import TalhaoDTO
from app.domains.talhoes.repository import TalhaoRepository
from app.domains.talhoes.schemas import (
    CreateTalhaoRequest,
    TalhaoResponse,
    UpdateTalhaoRequest,
)

if TYPE_CHECKING:
    from app.domains.fazendas.repository import FazendaRepository


class TalhaoService:
    def __init__(self, repo: TalhaoRepository, fazenda_repo: "FazendaRepository") -> None:
        self._repo = repo
        self._fazendas = fazenda_repo

    async def create(self, user_id: str, request: CreateTalhaoRequest) -> TalhaoResponse:
        fazenda_id = await self._resolve_fazenda(user_id, request.fazenda_id)
        talhao = await self._repo.create(user_id, fazenda_id, request)
        return self.to_response(talhao)

    async def list_for_user(
        self, user_id: str, fazenda_id: str | None = None
    ) -> list[TalhaoResponse]:
        if fazenda_id:
            talhoes = await self._repo.find_all_by_fazenda(fazenda_id, user_id)
        else:
            talhoes = await self._repo.find_all_by_user(user_id)
        return [self.to_response(t) for t in talhoes]

    async def update(
        self, talhao_id: str, user_id: str, request: UpdateTalhaoRequest
    ) -> TalhaoResponse:
        if request.fazenda_id is not None:
            await self._resolve_fazenda(user_id, request.fazenda_id)
        talhao = await self._repo.update(talhao_id, user_id, request)
        if not talhao:
            raise NotFoundError("Talhao", talhao_id)
        return self.to_response(talhao)

    async def delete(self, talhao_id: str, user_id: str) -> None:
        found = await self._repo.delete(talhao_id, user_id)
        if not found:
            raise NotFoundError("Talhao", talhao_id)

    async def _resolve_fazenda(self, user_id: str, fazenda_id: str | None) -> str:
        """TCC-096: a fazenda pedida precisa ser do usuário; sem pedido, a padrão."""
        if fazenda_id is None:
            return (await self._fazendas.get_or_create_default(user_id)).id
        if not await self._fazendas.find_by_id(fazenda_id, user_id):
            raise NotFoundError("Fazenda", fazenda_id)
        return fazenda_id

    @staticmethod
    def to_response(t: TalhaoDTO) -> TalhaoResponse:
        return TalhaoResponse(
            id=t.id,
            fazenda_id=t.fazenda_id,
            nome=t.nome,
            apelido=t.apelido,
            hectares=t.hectares,
            cultura=t.cultura,
            data_semeadura=t.data_semeadura,
            created_at=t.created_at,
        )
