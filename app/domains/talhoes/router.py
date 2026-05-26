from fastapi import APIRouter, Depends

from app.core.dependencies import get_current_user, get_talhao_service
from app.domains.auth.dto import UserDTO
from app.domains.talhoes.schemas import CreateTalhaoRequest, TalhaoResponse
from app.domains.talhoes.service import TalhaoService

router = APIRouter(prefix="/talhoes", tags=["Talhoes"])


@router.get("", response_model=list[TalhaoResponse])
async def list_talhoes(
    current_user: UserDTO = Depends(get_current_user),
    service: TalhaoService = Depends(get_talhao_service),
) -> list[TalhaoResponse]:
    return await service.list_for_user(current_user.id)


@router.post("", response_model=TalhaoResponse, status_code=201)
async def create_talhao(
    body: CreateTalhaoRequest,
    current_user: UserDTO = Depends(get_current_user),
    service: TalhaoService = Depends(get_talhao_service),
) -> TalhaoResponse:
    return await service.create(current_user.id, body)


@router.delete("/{talhao_id}", status_code=204)
async def delete_talhao(
    talhao_id: str,
    current_user: UserDTO = Depends(get_current_user),
    service: TalhaoService = Depends(get_talhao_service),
) -> None:
    await service.delete(talhao_id, current_user.id)
