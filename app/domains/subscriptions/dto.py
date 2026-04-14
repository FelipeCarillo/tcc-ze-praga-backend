from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class PlanDTO:
    id: str
    name: str
    display_name: str
    chat_daily_limit: int | None
    inference_daily_limit: int | None
    api_monthly_limit: int | None
    is_active: bool


@dataclass(frozen=True)
class SubscriptionDTO:
    id: str
    user_id: str
    plan: PlanDTO
    started_at: datetime
    expires_at: datetime | None
    is_active: bool
