from fastapi import APIRouter, Depends

from app.core.dependencies import get_auth_service, get_current_user
from app.domains.auth.dto import UserDTO
from app.domains.auth.schemas import LoginRequest, RegisterRequest, TokenResponse, UserResponse
from app.domains.auth.service import AuthService

router = APIRouter(prefix="/auth", tags=["Auth"])


@router.post("/register", response_model=TokenResponse, status_code=201)
async def register(
    body: RegisterRequest,
    service: AuthService = Depends(get_auth_service),
) -> TokenResponse:
    return await service.register(body)


@router.post("/login", response_model=TokenResponse)
async def login(
    body: LoginRequest,
    service: AuthService = Depends(get_auth_service),
) -> TokenResponse:
    return await service.login(body)


@router.get("/me", response_model=UserResponse)
async def me(current_user: UserDTO = Depends(get_current_user)) -> UserResponse:
    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        full_name=current_user.full_name,
        created_at=current_user.created_at,
    )
