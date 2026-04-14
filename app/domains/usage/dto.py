from dataclasses import dataclass
from datetime import datetime

from app.shared.enums import FeatureTypeEnum


@dataclass(frozen=True)
class FeatureUsageDTO:
    used: int
    limit: int | None  # None = unlimited


@dataclass(frozen=True)
class UsageSummaryDTO:
    chat: FeatureUsageDTO
    inference: FeatureUsageDTO
    api: FeatureUsageDTO


@dataclass(frozen=True)
class UsageLogDTO:
    id: str
    feature: FeatureTypeEnum
    used_at: datetime
    metadata: dict | None
