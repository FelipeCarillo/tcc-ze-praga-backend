from fastapi import APIRouter, Depends

from app.core.dependencies import get_current_user, get_fazenda_service
from app.domains.auth.dto import UserDTO
from app.domains.fazendas.schemas import (
    CreateFazendaRequest,
    FazendaResponse,
    UpdateFazendaRequest,
)
from app.domains.fazendas.service import FazendaService

router = APIRouter(prefix="/fazendas", tags=["Fazendas"])


@router.get("", response_model=list[FazendaResponse])
async def list_fazendas(
    current_user: UserDTO = Depends(get_current_user),
    service: FazendaService = Depends(get_fazenda_service),
) -> list[FazendaResponse]:
    """Fazendas do usuário, cada uma com os seus talhões."""
    return await service.list_for_user(current_user.id)


@router.post("", response_model=FazendaResponse, status_code=201)
async def create_fazenda(
    body: CreateFazendaRequest,
    current_user: UserDTO = Depends(get_current_user),
    service: FazendaService = Depends(get_fazenda_service),
) -> FazendaResponse:
    return await service.create(current_user.id, body)


@router.patch("/{fazenda_id}", response_model=FazendaResponse)
async def update_fazenda(
    fazenda_id: str,
    body: UpdateFazendaRequest,
    current_user: UserDTO = Depends(get_current_user),
    service: FazendaService = Depends(get_fazenda_service),
) -> FazendaResponse:
    return await service.update(fazenda_id, current_user.id, body)


@router.delete("/{fazenda_id}", status_code=204)
async def delete_fazenda(
    fazenda_id: str,
    current_user: UserDTO = Depends(get_current_user),
    service: FazendaService = Depends(get_fazenda_service),
) -> None:
    """Apaga a fazenda e os talhões dela; os laudos ficam sem talhão."""
    await service.delete(fazenda_id, current_user.id)
