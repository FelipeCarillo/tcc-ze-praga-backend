from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domains.auth.api_key_dto import ApiKeyCreateDTO, ApiKeyDTO
from app.models.api_key import ApiKey


class ApiKeyRepository:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def create(self, data: ApiKeyCreateDTO) -> ApiKeyDTO:
        row = ApiKey(
            user_id=data.user_id,
            name=data.name,
            key_hash=data.key_hash,
            key_prefix=data.key_prefix,
            scopes=list(data.scopes),
        )
        self._db.add(row)
        await self._db.commit()
        await self._db.refresh(row)
        return self._to_dto(row)

    async def find_active_by_user(self, user_id: str) -> list[ApiKeyDTO]:
        """Lista todas as keys do user, ativas E revogadas — ordenadas por created_at desc.

        Frontend mostra historico (incluindo revogadas) com o ``is_active`` flag,
        seguindo o padrao GitHub / Stripe.
        """
        result = await self._db.execute(
            select(ApiKey)
            .where(ApiKey.user_id == user_id)
            .order_by(ApiKey.created_at.desc())
        )
        rows = result.scalars().all()
        return [self._to_dto(r) for r in rows]

    async def find_by_id(self, key_id: str, user_id: str) -> ApiKeyDTO | None:
        """Busca por id escopado por user (evita IDOR)."""
        result = await self._db.execute(
            select(ApiKey).where(ApiKey.id == key_id, ApiKey.user_id == user_id)
        )
        row = result.scalar_one_or_none()
        return self._to_dto(row) if row else None

    async def find_by_prefix_active(self, prefix: str) -> list[ApiKeyDTO]:
        """Candidatas pra ``verify(plain_key)`` — apenas ativas, batem o prefix."""
        result = await self._db.execute(
            select(ApiKey).where(
                ApiKey.key_prefix == prefix,
                ApiKey.is_active == True,  # noqa: E712
            )
        )
        rows = result.scalars().all()
        return [self._to_dto(r) for r in rows]

    async def revoke(self, key_id: str, user_id: str) -> bool:
        """Marca ``is_active=False`` e seta ``revoked_at`` agora. Retorna True se afetou."""
        result = await self._db.execute(
            select(ApiKey).where(ApiKey.id == key_id, ApiKey.user_id == user_id)
        )
        row = result.scalar_one_or_none()
        if not row:
            return False
        row.is_active = False
        row.revoked_at = datetime.now(UTC)
        await self._db.commit()
        return True

    async def touch_last_used(self, key_id: str) -> None:
        result = await self._db.execute(select(ApiKey).where(ApiKey.id == key_id))
        row = result.scalar_one_or_none()
        if not row:
            return
        row.last_used_at = datetime.now(UTC)
        await self._db.commit()

    @staticmethod
    def _to_dto(row: ApiKey) -> ApiKeyDTO:
        return ApiKeyDTO(
            id=row.id,
            user_id=row.user_id,
            name=row.name,
            key_hash=row.key_hash,
            key_prefix=row.key_prefix,
            scopes=list(row.scopes or []),
            is_active=row.is_active,
            last_used_at=row.last_used_at,
            created_at=row.created_at,
            revoked_at=row.revoked_at,
        )
