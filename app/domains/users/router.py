from fastapi import APIRouter, Depends

from app.core.dependencies import get_current_user, get_user_service
from app.domains.auth.dto import UserDTO
from app.domains.users.schemas import UpdateUserRequest, UserProfileResponse
from app.domains.users.service import UserService

router = APIRouter(prefix="/users", tags=["Users"])


@router.get("/me", response_model=UserProfileResponse)
async def get_me(
    current_user: UserDTO = Depends(get_current_user),
    service: UserService = Depends(get_user_service),
) -> UserProfileResponse:
    return await service.get_profile(current_user.id)


@router.patch("/me", response_model=UserProfileResponse)
async def update_me(
    body: UpdateUserRequest,
    current_user: UserDTO = Depends(get_current_user),
    service: UserService = Depends(get_user_service),
) -> UserProfileResponse:
    return await service.update_profile(current_user.id, body)


@router.delete("/me", status_code=204)
async def delete_me(
    current_user: UserDTO = Depends(get_current_user),
    service: UserService = Depends(get_user_service),
) -> None:
    await service.delete_account(current_user.id)
