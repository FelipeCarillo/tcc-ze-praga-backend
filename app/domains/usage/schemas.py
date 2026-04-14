from datetime import datetime

from pydantic import BaseModel

from app.shared.enums import FeatureTypeEnum


class FeatureUsageResponse(BaseModel):
    used: int
    limit: int | None
    remaining: int | None

    @classmethod
    def from_counts(cls, used: int, limit: int | None) -> "FeatureUsageResponse":
        remaining = (limit - used) if limit is not None else None
        return cls(used=used, limit=limit, remaining=remaining)


class UsageSummaryResponse(BaseModel):
    chat: FeatureUsageResponse
    inference: FeatureUsageResponse
    api: FeatureUsageResponse


class UsageHistoryItemResponse(BaseModel):
    id: str
    feature: FeatureTypeEnum
    used_at: datetime
