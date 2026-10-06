from fastapi import APIRouter, Depends, Query

from app.core.dependencies import get_current_user, get_talhao_service
from app.domains.auth.dto import UserDTO
from app.domains.talhoes.schemas import (
    CreateTalhaoRequest,
    TalhaoResponse,
    UpdateTalhaoRequest,
)
from app.domains.talhoes.service import TalhaoService

router = APIRouter(prefix="/talhoes", tags=["Talhoes"])


@router.get("", response_model=list[TalhaoResponse])
async def list_talhoes(
    fazenda_id: str | None = Query(default=None, description="Só os talhões desta fazenda"),
    current_user: UserDTO = Depends(get_current_user),
    service: TalhaoService = Depends(get_talhao_service),
) -> list[TalhaoResponse]:
    return await service.list_for_user(current_user.id, fazenda_id)


@router.post("", response_model=TalhaoResponse, status_code=201)
async def create_talhao(
    body: CreateTalhaoRequest,
    current_user: UserDTO = Depends(get_current_user),
    service: TalhaoService = Depends(get_talhao_service),
) -> TalhaoResponse:
    return await service.create(current_user.id, body)


@router.patch("/{talhao_id}", response_model=TalhaoResponse)
async def update_talhao(
    talhao_id: str,
    body: UpdateTalhaoRequest,
    current_user: UserDTO = Depends(get_current_user),
    service: TalhaoService = Depends(get_talhao_service),
) -> TalhaoResponse:
    return await service.update(talhao_id, current_user.id, body)


@router.delete("/{talhao_id}", status_code=204)
async def delete_talhao(
    talhao_id: str,
    current_user: UserDTO = Depends(get_current_user),
    service: TalhaoService = Depends(get_talhao_service),
) -> None:
    await service.delete(talhao_id, current_user.id)
