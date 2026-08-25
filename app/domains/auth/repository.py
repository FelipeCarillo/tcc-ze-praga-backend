from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domains.auth.dto import EmailVerificationTokenDTO, UserCreateDTO, UserDTO
from app.models.email_verification_token import EmailVerificationToken
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

    async def create(self, data: UserCreateDTO, *, is_active: bool = True) -> UserDTO:
        user = User(
            email=data.email,
            password_hash=data.password_hash,
            full_name=data.full_name,
            is_active=is_active,
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


class EmailVerificationRepository:
    """Persistência dos tokens de verificação de e-mail (TCC-090)."""

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def create(
        self, user_id: str, token_hash: str, expires_at: datetime
    ) -> EmailVerificationTokenDTO:
        token = EmailVerificationToken(
            user_id=user_id, token_hash=token_hash, expires_at=expires_at
        )
        self._db.add(token)
        await self._db.commit()
        await self._db.refresh(token)
        return self._to_dto(token)

    async def find_by_hash(self, token_hash: str) -> EmailVerificationTokenDTO | None:
        result = await self._db.execute(
            select(EmailVerificationToken).where(EmailVerificationToken.token_hash == token_hash)
        )
        token = result.scalar_one_or_none()
        return self._to_dto(token) if token else None

    async def mark_used(self, token_id: str) -> None:
        result = await self._db.execute(
            select(EmailVerificationToken).where(EmailVerificationToken.id == token_id)
        )
        token = result.scalar_one_or_none()
        if not token:
            return
        token.used_at = datetime.now(UTC)
        await self._db.commit()

    async def invalidate_pending(self, user_id: str) -> None:
        """Queima os tokens ainda abertos do usuário.

        Chamado antes de emitir um novo (reenvio) — assim só o último link
        recebido funciona, e um e-mail antigo interceptado não serve mais.
        """
        result = await self._db.execute(
            select(EmailVerificationToken).where(
                EmailVerificationToken.user_id == user_id,
                EmailVerificationToken.used_at.is_(None),
            )
        )
        now = datetime.now(UTC)
        for token in result.scalars().all():
            token.used_at = now
        await self._db.commit()

    @staticmethod
    def _to_dto(token: EmailVerificationToken) -> EmailVerificationTokenDTO:
        return EmailVerificationTokenDTO(
            id=token.id,
            user_id=token.user_id,
            token_hash=token.token_hash,
            expires_at=token.expires_at,
            used_at=token.used_at,
            created_at=token.created_at,
        )
