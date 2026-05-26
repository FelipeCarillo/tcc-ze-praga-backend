from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domains.usage.dto import UsageLogDTO
from app.models.usage_log import UsageLog
from app.shared.enums import FeatureTypeEnum


class UsageRepository:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def count_today(self, user_id: str, feature: FeatureTypeEnum) -> int:
        today_start = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        result = await self._db.execute(
            select(func.count(UsageLog.id)).where(
                UsageLog.user_id == user_id,
                UsageLog.feature == feature,
                UsageLog.used_at >= today_start,
            )
        )
        return result.scalar_one()

    async def count_this_month(self, user_id: str, feature: FeatureTypeEnum) -> int:
        now = datetime.now(UTC)
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        result = await self._db.execute(
            select(func.count(UsageLog.id)).where(
                UsageLog.user_id == user_id,
                UsageLog.feature == feature,
                UsageLog.used_at >= month_start,
            )
        )
        return result.scalar_one()

    async def record(
        self,
        user_id: str,
        feature: FeatureTypeEnum,
        metadata: dict[str, Any] | None = None,
    ) -> UsageLogDTO:
        log = UsageLog(user_id=user_id, feature=feature, metadata_=metadata)
        self._db.add(log)
        await self._db.commit()
        await self._db.refresh(log)
        return self._to_dto(log)

    async def find_recent(self, user_id: str, limit: int = 50) -> list[UsageLogDTO]:
        result = await self._db.execute(
            select(UsageLog)
            .where(UsageLog.user_id == user_id)
            .order_by(UsageLog.used_at.desc())
            .limit(limit)
        )
        return [self._to_dto(log) for log in result.scalars().all()]

    @staticmethod
    def _to_dto(log: UsageLog) -> UsageLogDTO:
        return UsageLogDTO(
            id=log.id,
            feature=FeatureTypeEnum(log.feature),
            used_at=log.used_at,
            metadata=log.metadata_,
        )
