from app.core.exceptions import ConflictError, UnauthorizedError
from app.core.security import create_access_token, hash_password, verify_password
from app.domains.auth.dto import UserCreateDTO, UserDTO
from app.domains.auth.repository import UserRepository
from app.domains.auth.schemas import LoginRequest, RegisterRequest, TokenResponse, UserResponse


class AuthService:
    def __init__(self, user_repo: UserRepository) -> None:
        self._user_repo = user_repo

    async def register(self, request: RegisterRequest) -> TokenResponse:
        if await self._user_repo.find_by_email(request.email):
            raise ConflictError("Email already in use")

        hashed = hash_password(request.password)
        user = await self._user_repo.create(
            UserCreateDTO(
                email=request.email,
                password_hash=hashed,
                full_name=request.full_name,
            )
        )
        return self._build_token_response(user)

    async def login(self, request: LoginRequest) -> TokenResponse:
        user = await self._user_repo.find_by_email(request.email)
        if not user or not verify_password(request.password, user.password_hash):
            raise UnauthorizedError("Invalid email or password")
        if not user.is_active:
            raise UnauthorizedError("Account is inactive")
        return self._build_token_response(user)

    @staticmethod
    def _build_token_response(user: UserDTO) -> TokenResponse:
        token = create_access_token(user.id)
        return TokenResponse(
            access_token=token,
            user=UserResponse(
                id=user.id,
                email=user.email,
                full_name=user.full_name,
                created_at=user.created_at,
            ),
        )
