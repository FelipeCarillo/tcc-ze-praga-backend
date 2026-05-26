from app.core.exceptions import NotFoundError
from app.domains.talhoes.dto import TalhaoDTO
from app.domains.talhoes.repository import TalhaoRepository
from app.domains.talhoes.schemas import CreateTalhaoRequest, TalhaoResponse


class TalhaoService:
    def __init__(self, repo: TalhaoRepository) -> None:
        self._repo = repo

    async def create(self, user_id: str, request: CreateTalhaoRequest) -> TalhaoResponse:
        talhao = await self._repo.create(user_id, request)
        return self._to_response(talhao)

    async def list_for_user(self, user_id: str) -> list[TalhaoResponse]:
        talhoes = await self._repo.find_all_by_user(user_id)
        return [self._to_response(t) for t in talhoes]

    async def delete(self, talhao_id: str, user_id: str) -> None:
        found = await self._repo.delete(talhao_id, user_id)
        if not found:
            raise NotFoundError("Talhao", talhao_id)

    @staticmethod
    def _to_response(t: TalhaoDTO) -> TalhaoResponse:
        return TalhaoResponse(
            id=t.id,
            nome=t.nome,
            apelido=t.apelido,
            hectares=t.hectares,
            cultura=t.cultura,
            data_semeadura=t.data_semeadura,
            created_at=t.created_at,
        )
