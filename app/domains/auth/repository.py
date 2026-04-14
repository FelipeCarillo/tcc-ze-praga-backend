from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domains.auth.dto import UserCreateDTO, UserDTO
from app.models.user import User


class UserRepository:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def find_by_email(self, email: str) -> UserDTO | None:
        result = await self._db.execute(select(User).where(User.email == email))
        user = result.scalar_one_or_none()
        return self._to_dto(user) if user else None

    async def find_by_id(self, user_id: str) -> UserDTO | None:
        result = await self._db.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()
        return self._to_dto(user) if user else None

    async def create(self, data: UserCreateDTO) -> UserDTO:
        user = User(
            email=data.email,
            password_hash=data.password_hash,
            full_name=data.full_name,
        )
        self._db.add(user)
        await self._db.commit()
        await self._db.refresh(user)
        return self._to_dto(user)

    async def update(self, user_id: str, **fields: object) -> UserDTO | None:
        result = await self._db.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()
        if not user:
            return None
        for key, value in fields.items():
            setattr(user, key, value)
        await self._db.commit()
        await self._db.refresh(user)
        return self._to_dto(user)

    async def soft_delete(self, user_id: str) -> None:
        await self.update(user_id, is_active=False)

    @staticmethod
    def _to_dto(user: User) -> UserDTO:
        return UserDTO(
            id=user.id,
            email=user.email,
            password_hash=user.password_hash,
            full_name=user.full_name,
            is_active=user.is_active,
            created_at=user.created_at,
            updated_at=user.updated_at,
        )
