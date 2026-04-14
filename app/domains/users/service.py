from app.core.exceptions import ConflictError, NotFoundError
from app.domains.auth.dto import UserDTO
from app.domains.auth.repository import UserRepository
from app.domains.users.schemas import UpdateUserRequest, UserProfileResponse


class UserService:
    def __init__(self, user_repo: UserRepository) -> None:
        self._user_repo = user_repo

    async def get_profile(self, user_id: str) -> UserProfileResponse:
        user = await self._user_repo.find_by_id(user_id)
        if not user:
            raise NotFoundError("User", user_id)
        return self._to_profile(user)

    async def update_profile(self, user_id: str, request: UpdateUserRequest) -> UserProfileResponse:
        if request.email:
            existing = await self._user_repo.find_by_email(request.email)
            if existing and existing.id != user_id:
                raise ConflictError("Email already in use")

        updates = request.model_dump(exclude_none=True)
        user = await self._user_repo.update(user_id, **updates)
        if not user:
            raise NotFoundError("User", user_id)
        return self._to_profile(user)

    async def delete_account(self, user_id: str) -> None:
        await self._user_repo.soft_delete(user_id)

    @staticmethod
    def _to_profile(user: UserDTO) -> UserProfileResponse:
        return UserProfileResponse(
            id=user.id,
            email=user.email,
            full_name=user.full_name,
            is_active=user.is_active,
            created_at=user.created_at,
            updated_at=user.updated_at,
        )
