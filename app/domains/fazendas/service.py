from app.core.exceptions import NotFoundError
from app.domains.fazendas.dto import FazendaDTO
from app.domains.fazendas.repository import FazendaRepository
from app.domains.fazendas.schemas import (
    CreateFazendaRequest,
    FazendaResponse,
    UpdateFazendaRequest,
)
from app.domains.talhoes.repository import TalhaoRepository
from app.domains.talhoes.schemas import TalhaoResponse
from app.domains.talhoes.service import TalhaoService


class FazendaService:
    """Fazendas do produtor, cada uma com os seus talhões (TCC-096)."""

    def __init__(self, repo: FazendaRepository, talhao_repo: TalhaoRepository) -> None:
        self._repo = repo
        self._talhoes = talhao_repo

    async def list_for_user(self, user_id: str) -> list[FazendaResponse]:
        """Lista as fazendas já com os talhões aninhados — a hierarquia da UI."""
        fazendas = await self._repo.find_all_by_user(user_id)
        talhoes = await self._talhoes.find_all_by_user(user_id)
        por_fazenda: dict[str, list[TalhaoResponse]] = {}
        for t in talhoes:
            por_fazenda.setdefault(t.fazenda_id, []).append(TalhaoService.to_response(t))
        return [self._to_response(f, por_fazenda.get(f.id, [])) for f in fazendas]

    async def create(self, user_id: str, request: CreateFazendaRequest) -> FazendaResponse:
        return self._to_response(await self._repo.create(user_id, request), [])

    async def update(
        self, fazenda_id: str, user_id: str, request: UpdateFazendaRequest
    ) -> FazendaResponse:
        fazenda = await self._repo.update(fazenda_id, user_id, request)
        if not fazenda:
            raise NotFoundError("Fazenda", fazenda_id)
        talhoes = await self._talhoes.find_all_by_fazenda(fazenda_id, user_id)
        return self._to_response(fazenda, [TalhaoService.to_response(t) for t in talhoes])

    async def delete(self, fazenda_id: str, user_id: str) -> None:
        if not await self._repo.delete(fazenda_id, user_id):
            raise NotFoundError("Fazenda", fazenda_id)

    @staticmethod
    def _to_response(f: FazendaDTO, talhoes: list[TalhaoResponse]) -> FazendaResponse:
        return FazendaResponse(
            id=f.id,
            nome=f.nome,
            municipio=f.municipio,
            uf=f.uf,
            hectares=f.hectares,
            agronomo_nome=f.agronomo_nome,
            agronomo_crea=f.agronomo_crea,
            created_at=f.created_at,
            talhoes=talhoes,
        )
